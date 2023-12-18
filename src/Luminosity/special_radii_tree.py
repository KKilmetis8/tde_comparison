#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 16:02:41 2023

@author: konstantinos, Paola

NOTES FOR OTHERS:
- things from snapshots are in solar and code units (mass in M_sol, 
  length in R_sol, time s.t. G=1), we have to convert them in CGS 

- change m, fixes, loadpath
"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

# Vanilla Imports
import numpy as np
import healpy as hp
from scipy.stats import gmean

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from datetime import datetime
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [5 , 3]
plt.rcParams['axes.facecolor'] = 'whitesmoke'

# Custom Imports
from src.Opacity.opacity_table import opacity
from src.Calculators.ray_tree import ray_maker
from src.Luminosity.select_path import select_snap

Rsol_to_cm = 6.957e10 # [cm]


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

    Returns
    -------
    kappar: float.
            The optical depth of a cell.
    '''    
    # If there is nothing, the ray continues unimpeded
    if rho < np.exp(-49.3):        
        return 0
    
    # Stream material, is opaque
    elif T < np.exp(8.666):
        return 100
    
    # Too hot: scale as Kramers for absorption (planck)
    elif T > np.exp(17.876):
        # X = 0.7389
        # Z = 0.02
        
        # Constant value for scatter
        Tmax = np.exp(17.87)
        kscattering = opacity(Tmax, rho, 'scattering', ln = False)
        
        # Scale as Kramers the last point for absorption
        kplank_0 = opacity(Tmax, rho, 'planck', ln = False)
        kplanck = kplank_0 * (T/Tmax)**(-3.5)
        # kplanck =  3.8e22 * (1 + X) * T**(-3.5) * rho # [cm^2/g] Kramers'law
        # kplanck *= rho

        if select == 'photo':
            k = kplanck + kscattering
        
        if select == 'thermr':
            k = np.sqrt(3 * kplanck * (kplanck + kscattering)) 

        kappa_high = k * r_dlogr
        return kappa_high 
    
    else:
        # Lookup table
        if select == 'photo':
            k = opacity(T, rho,'red', ln = False)
        if select == 'thermr':
            k = opacity(T, rho,'effective', ln = False)

        kappa =  k * r_dlogr

        return kappa

def calc_specialr(T, rho, radius, branch_indexes, select):
    '''
    Finds and saves the photosphere/R_therm (CGS) for ONE  ray.

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

    kappa = 0
    kappas = []
    cumulative_kappas = []
    i = -1 # Initialize reverse loop
    while kappa <= threshold and i > -len(T):
        dlogr = np.log(radius[i]) - np.log(radius[i-1]) #back to logspace
        r_dlogr = radius[i] * dlogr #to integrate in log space
        new_kappa = get_kappa(T[i], rho[i], r_dlogr, select)
        kappa += new_kappa
        kappas.append(new_kappa)
        cumulative_kappas.append(kappa)
        i -= 1

    specialr =  radius[i] #i it's negative
    index_specialr = i + len(T) 
    branch_index_specialr = branch_indexes[i]

    return kappas, cumulative_kappas, specialr, index_specialr, branch_index_specialr

def get_specialr(rays_T, rays_den, radius, tree_indexes, select):
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
    plot = False
    check = 'fid'
    m = 6 

    now = datetime.now()
    now = now.strftime("%d/%m/%Y %H:%M:%S")
    snapshots, days = select_snap(m, check)

    fix_photo_arit = np.zeros(len(snapshots))
    fix_photo_geom = np.zeros(len(snapshots))
    fix_thermr_arit = np.zeros(len(snapshots))
    fix_thermr_geom = np.zeros(len(snapshots))

    for index in range(0,len(snapshots)):        
        print('Snapshot ' + str(snapshots[index]))
        tree_indexes, rays_T, rays_den, rays, radii, rays_vol = ray_maker(snapshots[index], m, check)

        # dim_ph = np.zeros(len(rays_index_photo))
        # sushi = np.zeros(192)
        # for j in range(len(rays_index_photo)):
        #     find_index_cell = int(rays_index_photo[j])
        #     vol_ph = rays_vol[j][find_index_cell]
        #     dim_ph[j] = (3 * vol_ph /(4 * np.pi))**(1/3) #in solar units
        #     dim_grid = (radii[find_index_cell+1]-radii[find_index_cell])/Rsol_to_cm #in solar units
        #     sushi[j] = dim_ph[j] / dim_grid
        #     print('Simulation cell R: ' + str(dim_ph[j]))
        #     print('Our grid: ' + str(dim_grid))
        # sushi_mean = np.mean(sushi)
        # print('ratio: ' + str(sushi_mean))

        if photosphere:
            rays_kappa, rays_cumulative_kappas, rays_photo, _, _ = get_specialr(rays_T, rays_den, radii, tree_indexes, select='photo')
            rays_photo = rays_photo/Rsol_to_cm # to solar unit to plot

            fix_photo_arit[index] = np.mean(rays_photo)
            fix_photo_geom[index] = gmean(rays_photo)

            if plot:
                fig, ax = plt.subplots(figsize = (8,6))
                img = ax.scatter(np.arange(192), rays_photo, c = 'k', s = 15)
                plt.axhline(np.mean(rays_photo), c = 'r', linestyle = '--', label = r'$\bar{R}_{ph}$ arit mean')
                plt.axhline(gmean(rays_photo), c = 'b', linestyle = '--', label = r'$\bar{R}_{ph}$ geom mean')
                # plt.axhline(800, c = 'r', label = r'$\bar{R}_{ph}$ arit mean') #Elad
                # plt.axhline(50, c = 'b', label = r'$\bar{R}_{ph}$ geom mean') #Elad
                plt.xlabel('Observers')
                plt.ylabel('$\log_{10} R_{ph} [R_\odot]$')
                plt.yscale('log')
                plt.grid()
                plt.legend()
                plt.savefig('photo_obs' + str(snapshots[index]) + '.png')
                plt.show()  
                

        if thermalisation: 
            rays_tau, rays_cumulative_taus, rays_thermr, _, _ = get_specialr(rays_T, rays_den, radii, tree_indexes, select= 'thermr')
            rays_thermr = rays_thermr/Rsol_to_cm # to solar unit to plot

            fix_thermr_arit[index] = np.mean(rays_thermr)
            fix_thermr_geom[index] = gmean(rays_thermr)
            
            if plot: 
                fig, ax = plt.subplots(figsize = (8,6))
                img = ax.scatter(np.arange(192), rays_thermr, c = 'k', s = 15)
                plt.axhline(np.mean(rays_thermr), c = 'r', linestyle = '--', label = r'$\bar{R}_{ph}$ arit mean')
                plt.axhline(gmean(rays_thermr), c = 'b', linestyle = '--', label = r'$\bar{R}_{ph}$ geom mean')
                # plt.axhline(800, c = 'r', label = r'$\bar{R}_{ph}$ arit mean') #Elad
                # plt.axhline(50, c = 'b', label = r'$\bar{R}_{ph}$ geom mean') #Elad
                plt.xlabel('Observers')
                plt.ylabel('$\log_{10} R_{therm} [R_\odot]$')
                plt.yscale('log')
                plt.grid()
                plt.legend()
                plt.savefig('therm_obs' + str(snapshots[index]) + '.png')
                plt.show()   


    if photosphere:         
        with open(f'data/special_radii_m{m}.txt', 'a') as file:
            file.write('# Run of ' + now + '\n#t/t_fb\n')
            file.write(' '.join(map(str, days)) + '\n')
            file.write('# Photosphere arithmetic mean \n')
            file.write(' '.join(map(str, fix_photo_arit)) + '\n')
            file.write('# Photosphere geometric mean \n')
            file.write(' '.join(map(str, fix_photo_geom)) + '\n')
            file.close()
            
    if thermalisation:
        with open(f'data/special_radii_m{m}.txt', 'a') as file:
            file.write('# Run of ' + now + '\n#t/t_fb\n')
            file.write(' '.join(map(str, days)) + '\n')
            file.write('# Thermalisation radius arithmetic mean \n')
            file.write(' '.join(map(str, fix_thermr_arit)) + '\n')
            file.write('# Thermalisation radius geometric mean \n')
            file.write(' '.join(map(str, fix_thermr_geom)) + '\n')
            file.close()



            