#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 16:02:41 2023

@author: konstantinos, Paola

NOTES FOR OTHERS:
- things from snapshots are in solar and code units (mass in M_sol, 
  length in R_sol, time s.t. G=1), we have to convert them in CGS 

- change m
"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

from src.Utilities.isalice import isalice
alice, plot = isalice()

# Vanilla Imports
import numpy as np
from scipy.stats import gmean
import h5py
import healpy as hp

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from datetime import datetime
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [5 , 3]
plt.rcParams['axes.facecolor'] = 'whitesmoke'
NSIDE = 4

# Custom Imports
from src.Opacity.cloudy_opacity import old_opacity 
from src.Calculators.ray_forest import find_sph_coord, ray_maker_forest
from src.Luminosity.select_path import select_snap

Rsol_to_cm = 7e10 #6.957e10 # [cm]


################
# FUNCTIONS
################

def get_kappa(T: float, rho: float, r_dlogr: float, select: str):
    '''
    Calculates the integrand of eq.(8) or (9) Steinberg&Stone22.

    Parameters
    ----------
    T: int.
        Cell temperature (CGS).
    rho: int.
        Cell density (CGS).
    r_dlogr: int.
            Deltar to integrate in logspace (CGS).
    select: str.
            Choose if you want photosphere o Rtherm.

    Returns
    -------
    kappa: float.
            The optical depth of a cell.
    '''    
    Tmax = 1e13 
    Tmin = 316
    # If there is nothing, the ray continues unimpeded
    if rho < np.exp(-49.3):
        print('rho low')        
        return 0
    
    # Stream material, is opaque
    elif T < Tmin:
        # print('T low')
        return 100
    
    # Too hot: scale as Kramers for absorption (planck)
    elif T > Tmax:
        # print('T high')
        # X = 0.7389
        # Z = 0.02
        
        # Constant value for scatter
        kscattering = old_opacity(Tmax, rho, 'scattering') 
        
        # Scale as Kramers the last point for absorption
        kplank_0 = old_opacity(Tmax, rho, 'planck')
        kplanck = kplank_0 * (T/Tmax)**(-3.5)
        # kplanck =  3.8e22 * (1 + X) * T**(-3.5) * rho # [cm^2/g] Kramers'law
        # kplanck *= rho

        if select == 'photo':
            k = kplanck + kscattering
        
        if select == 'thermr' or select == 'thermr_plot':
            k = np.sqrt(3 * kplanck * (kplanck + kscattering)) 

        kappa_high = k * r_dlogr

        return kappa_high 
    
    else:
        # Lookup table
        if select == 'photo':
            k = old_opacity(T, rho,'red')

        if select == 'thermr' or select == 'thermr_plot':
            k = old_opacity(T, rho,'effective')

        kappa =  k * r_dlogr

        return kappa

def calc_specialr(T, rho, radius, branch_indexes, select):
    '''
    Finds and saves the photosphere or R_therm (CGS) for ONE  ray.

    Parameters
    ----------
    T: 1D arrays.
            Temperature of every cell in a ray (CGS).
    rho: 1D arrays.
            Density of every cell in a ray (CGS).
    radius: 1D array.
            Radius (CGS).
    branch_indexes: 1D array.
                    Tree indexes for cells in the ray.
    select: str.
            Choose if you want photosphere o Rtherm.

    Returns
    -------
    kappas: 1D array.
                The "optical depth" of a single cell. 
    cumulative_kappas: 1D array.
                The total "optical depth" of a single cell.
    specialr: int.
           R_photosphere/R_therm (CGS) 
    index_specialr: int
            photosphere/R_therm index in our radius.
    branch_index_specialr: int.
                    Photosphere/R_therm index in the tree.
    '''
    if select == 'photo':
        threshold = 2/3
    if select == 'thermr': 
        threshold = 5
    if select == 'thermr_plot': 
        # to have the plot of extended figure 9 from Steinberg&Stone22
        threshold = 1

    kappa = 0
    kappas = []
    cumulative_kappas = []
    i = -1 # Initialize reverse loop
    while kappa <= threshold and i > -len(T):
        dlogr = np.log(radius[i]) - np.log(radius[i-1]) #back to logspace (ln NOT log10)
        r_dlogr = radius[i] * dlogr #to integrate in ln space
        new_kappa = get_kappa(T[i], rho[i], r_dlogr, select)
        kappa += new_kappa
        kappas.append(new_kappa)
        cumulative_kappas.append(kappa)
        i -= 1
        
    specialr =  radius[i] #i it's negative
    index_specialr = i + len(T) 
    branch_index_specialr = branch_indexes[i]

    return kappas, cumulative_kappas, specialr, index_specialr, branch_index_specialr

def get_specialr(rays_T, rays_den, radii, tree_indexes, select):
    '''
    Finds and saves the photosphere/R_therm (CGS) at every ray.

    Parameters
    ----------
    rays_T: nD arrays.
            Temperature of every ray/cell (CGS).
    rays_den: nD arrays.
            Density of every ray/cell (CGS).
    radius: 1D array.
            Radius (CGS).
    tree_indexes: nD array.
                Tree indexes for cells in the rays.
    select: str.
        Choose if you want photosphere o Rtherm.

    Returns
    -------
    rays_kappas: nD array.
                The "optical depth" of a single cell in every ray. 
    rays_cumulative_kappas: nD array.
                The total "optical depth" of a single cell in every ray.
    rays_specialr: 1D array.
                Photosphere/R_therm in every ray (CGS).
    rays_index_specialr: 1D array.
                     Photosphere/R_therm index in our radius for every ray.
    tree_index_specialr: 1D array.
                    Photosphere/R_therm index in the tree for every ray.
    '''
    rays_kappas = []
    rays_cumulative_kappas = []
    rays_specialr = np.zeros(len(rays_T))
    rays_index_specialr = np.zeros(len(rays_T))
    tree_index_specialr = np.zeros(len(rays_T))
    
    for i in range(len(rays_T)):

        # Isolate each ray
        T_of_single_ray = rays_T[i]
        Den_of_single_ray = rays_den[i]
        branch_indexes = tree_indexes[i]
        radius = radii[i]
        
        # Get photosphere/R_therm
        kappas, cumulative_kappas, specialr, index_ph, branch_index_ph  = calc_specialr(T_of_single_ray, Den_of_single_ray, 
                                                                       radius, branch_indexes, select)
        # Store
        rays_kappas.append(kappas)
        rays_cumulative_kappas.append(cumulative_kappas)
        rays_specialr[i] = specialr
        rays_index_specialr[i] = index_ph
        tree_index_specialr[i] = branch_index_ph

    return rays_kappas, rays_cumulative_kappas, rays_specialr, rays_index_specialr, tree_index_specialr


################
# MAIN
################

if __name__ == "__main__":
    photosphere = True
    thermalisation = True
    save = True
    check = 'fid'
    m = 6 
    num = 1000

    now = datetime.now()
    now = now.strftime("%d/%m/%Y %H:%M:%S")
    snapshots, days = select_snap(m, check)

    fix_photo_arit = np.zeros(len(snapshots))
    fix_photo_geom = np.zeros(len(snapshots))
    fix_thermr_arit = np.zeros(len(snapshots))
    fix_thermr_geom = np.zeros(len(snapshots))

    for index in range(len(snapshots)): 
        snap = snapshots[index]       
        print('Snapshot ' + str(snap))
        filename = f"{m}/{snap}/snap_{snap}.h5"

        box = np.zeros(6)
        with h5py.File(filename, 'r') as fileh:
            for i in range(len(box)):
                box[i] = fileh['Box'][i]
        # print('Box', box)

        # Find observers with Healpix 
        # For each of them set the upper limit for R
        thetas = np.zeros(192)
        phis = np.zeros(192) 
        observers = []
        stops = np.zeros(192) 
        xyz_grid = []
        for iobs in range(0,192):
            theta, phi = hp.pix2ang(NSIDE, iobs) # theta in [0,pi], phi in [0,2pi]
            thetas[iobs] = theta
            phis[iobs] = phi
            observers.append( (theta, phi) )
            xyz = find_sph_coord(1, theta, phi)
            xyz_grid.append(xyz)

            mu_x = xyz[0]
            mu_y = xyz[1]
            mu_z = xyz[2]

            # Box is for 
            if(mu_x < 0):
                rmax = box[0] / mu_x
            else:
                rmax = box[3] / mu_x
            if(mu_y < 0):
                rmax = min(rmax, box[1] / mu_y)
            else:
                rmax = min(rmax, box[4] / mu_y)
            if(mu_z < 0):
                rmax = min(rmax, box[2] / mu_z)
            else:
                rmax = min(rmax, box[5] / mu_z)

            stops[iobs] = rmax

        # rays is Er of Elad 
        tree_indexes, rays_T, rays_den, rays, rays_ie, rays_radii, _, rays_v = ray_maker_forest(snap, m, check, thetas, phis, stops, num)

        if photosphere:
            rays_kappa, rays_cumulative_kappas, rays_photo, _, _ = get_specialr(rays_T, rays_den, rays_radii, tree_indexes, select='photo')
            rays_photo = rays_photo/Rsol_to_cm # to solar unit to plot

            fix_photo_arit[index] = np.mean(rays_photo)
            fix_photo_geom[index] = gmean(rays_photo)
           

        if thermalisation: 
            rays_tau, rays_cumulative_taus, rays_thermr, _, _ = get_specialr(rays_T, rays_den, rays_radii, tree_indexes, select= 'thermr_plot')
            rays_thermr = rays_thermr/Rsol_to_cm # to solar unit to plot

            fix_thermr_arit[index] = np.mean(rays_thermr)
            fix_thermr_geom[index] = gmean(rays_thermr)
            

    if save: 
        if alice:
            pre_saving = '/home/s3745597/data1/TDE/tde_comparison/data/'
        else:
            pre_saving = 'data/'

        if photosphere:         
            with open(f'{pre_saving}DYNspecial_radii_m{m}_box.txt', 'a') as file:
                file.write('# Run of ' + now + ' with CLOUDY opacity \n#t/t_fb\n')
                file.write(' '.join(map(str, days)) + '\n')
                file.write('# Photosphere arithmetic mean \n')
                file.write(' '.join(map(str, fix_photo_arit)) + '\n')
                file.write('# Photosphere geometric mean \n')
                file.write(' '.join(map(str, fix_photo_geom)) + '\n')
                file.close()
                
        if thermalisation:
            with open(f'data/DYNspecial_radii_m{m}_box.txt', 'a') as file:
                file.write('# Run of ' + now + ' with CLOUDY opacity \n#t/t_fb\n')
                file.write(' '.join(map(str, days)) + '\n')
                file.write('# Thermalisation radius arithmetic mean \n')
                file.write(' '.join(map(str, fix_thermr_arit)) + '\n')
                file.write('# Thermalisation radius geometric mean \n')
                file.write(' '.join(map(str, fix_thermr_geom)) + '\n')
                file.close()


