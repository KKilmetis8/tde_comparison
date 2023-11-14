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
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [5 , 3]
plt.rcParams['axes.facecolor'] = 'whitesmoke'

# Custom Imports
from src.Opacity.opacity_table import opacity
from src.Calculators.ray_tree import ray_maker

################
# FUNCTIONS
################

def select_fix(m):
    if m == 4:
        snapshots = [233] #, 254, 263, 277 , 293, 308, 322]
        days = [1]# , 1.2, 1.3, 1.4, 1.56, 1.7, 1.8] 
    if m == 6:
        snapshots = [844, 881, 925, 950] 
        days = [1, 1.1, 1.3, 1.4] 
    return snapshots, days

def get_kappa(T: float, rho: float, dr: float):
    '''
    Calculates the integrand of eq.(8) Steinberg&Stone22.
    '''    
    # If there is nothing, the ray continues unimpeded
    if rho < np.exp(-49.3):
        return 0
    
    # Stream material, is opaque
    if T < np.exp(8.666):
        return 100
    
    # Too hot: Kramers' law for absorption (planck)
    if T > np.exp(17.876):
        X = 0.7389
        Z = 0.02
        # kplanck =  1.2e26 * Z * (1 + X) * T**(-3.5) * rho #Kramers' bound-free opacity [cm^2/g]
        kplanck = 3.68e22 * (1-Z) * (1 + X) * T**(-3.5) * rho #Kramers' free-free opacity [cm^2/g]
        kplanck *= rho

        Tscatter = np.exp(17.87)
        kscattering = opacity(Tscatter, rho, 'scattering', ln = False)

        oppi = kplanck + kscattering
        tau_high = oppi * dr
        return tau_high 
    
    # Lookup table
    k = opacity(T, rho,'red', ln = False)
    kappar =  k * dr
    
    return kappar

def calc_photosphere(T, rho, rs):
    '''
    Input: 1D arrays.
    Finds and saves the photosphere (in solar units).
    The kappas' arrays go from far to near the BH.
    '''
    threshold = 2/3
    kappa = 0
    kappas = []
    cumulative_kappas = []
    i = -1 # Initialize reverse loop
    while kappa <= threshold and i > -len(T):
        dr = rs[i]-rs[i-1] # Cell seperation
        new_kappa = get_kappa(T[i], rho[i], dr)
        kappa += new_kappa
        kappas.append(new_kappa)
        cumulative_kappas.append(kappa)
        i -= 1

    photo =  rs[i] #i it's negative
    index_ph = len(rs) + i # to find the index of photo from inside to outside

    return kappas, cumulative_kappas, photo, index_ph

def get_photosphere(rays_T, rays_den, radii):
    '''
    Finds and saves the photosphere (in solar units) for every ray.

    Parameters
    ----------
    rays_T, rays_den: n-D arrays.
    radii: 1D array.

    Returns
    -------
    rays_kappas, rays_cumulative_kappas: nD arrays.
    photos: 1D array.
    '''
    # Get the thermalisation radius
    rays_kappas = []
    rays_cumulative_kappas = []
    rays_photo = np.zeros(len(rays_T))
    rays_index_photo = np.zeros(len(rays_T))
    
    for i in range(len(rays_T)):
        # Isolate each ray
        T_of_single_ray = rays_T[i]
        Den_of_single_ray = rays_den[i]
        
        # Get photosphere
        kappas, cumulative_kappas, photo, index_ph  = calc_photosphere(T_of_single_ray, Den_of_single_ray, radii)

        # Store
        rays_kappas.append(kappas)
        rays_cumulative_kappas.append(cumulative_kappas)
        rays_photo[i] = photo
        rays_index_photo[i] = index_ph

    return rays_kappas, rays_cumulative_kappas, rays_photo, rays_index_photo


def optical_depth(T, rho, dr):
    '''
    Calculates the optical depth at a point

    Parameters
    ----------
    T : float,
        Temperature in [cgs]. 
    rho : float. 
        Density in [cgs]. 

    dr : float,
        Cell Size in R_sol.

    Returns
    -------
    tau : float,
        The optical depth in [cgs].
    '''    
    # If there is nothing, the ray continues unimpeded
    if rho < np.exp(-49.3):
        #print('rho small')
        return 0
    
    # Stream material, is opaque
    if T < np.exp(8.666):
        # print('T low')
        return 100
    
    # Too hot: Kramers' law
    if T > np.exp(17.876):
        # T = np.exp(17.87)
        X = 0.7389
        Z = 0.02
        kplanck =  1.2e26 * Z * (1 + X) * T**(-3.5) * rho #Kramers' bound-free opacity [cm^2/g]
        kplanck *= rho
        
        Tscatter = np.exp(17.87)
        kscattering = opacity(Tscatter, rho,'scattering', ln = False)
        oppi = np.sqrt(3 * kplanck * (kplanck + kscattering)) 
        tau_high = oppi * dr
        
        return tau_high 
    
    # Lookup table
    oppi = opacity(T, rho,'effective', ln = False)
    tau =  oppi * dr
    
    return tau

def calc_thermr(rs, T, rho, threshold = 1):
    '''
    Finds and saves the effective optical depth at every cell the ray passess through.
    We use it to find the thermr of the ray.

    Parameters
    ----------
    rs : arr
        Radial coordinates of a ray
    rho : arr,
        Densities in a ray.
    T : arr,
        Temperatures in a ray
    threshold : float, optional
        The desired optical depth. The default is 1.

    Returns
    -------
    taus : np.array,
        The optical depth of a single cell.
        
    thermr : float,
        Where the thermr is for that ray.

    cumulative_taus : np.array,
        The total optical depth of a single cell.
    '''
    tau = 0
    taus = []
    cumulative_taus = []
    i = -1 # Initialize reverse loop
    while tau < threshold and i > -len(T):
        dr = rs[i]-rs[i-1] # Cell seperation
        new_tau = optical_depth(T[i], rho[i], dr)
        tau += new_tau
        taus.append(new_tau)
        cumulative_taus.append(tau)
        i -= 1
    thermr =  rs[i] #i it's negative
    return taus, thermr, cumulative_taus

def get_thermr(rays_T, rays_den, radii):
    # Get the thermr for every observer
    rays_tau = []
    rays_cumulative_taus = []
    thermr = np.zeros(len(rays_T))
    
    for i in range(len(rays_T)):
        
        # Isolate each ray
        T_of_single_ray = rays_T[i]
        Den_of_single_ray = rays_den[i]
        
        # Get thermr
        taus, th, cumulative_taus = calc_thermr(radii, T_of_single_ray, Den_of_single_ray, 
                                        threshold = 1)
        # Store
        rays_tau.append(taus)
        rays_cumulative_taus.append(cumulative_taus)
        thermr[i] = th

    return rays_tau, thermr, rays_cumulative_taus


################
# MAIN
################

if __name__ == "__main__":
    plot_ph = True
    plot_radii = True 
    m = 6 
    loadpath = str(m) + '/'

    snapshots, days = select_fix(m)
    fix_photo_arit = np.zeros(len(snapshots))
    fix_photo_geom = np.zeros(len(snapshots))
    
    fix_thermr_arit = np.zeros(len(snapshots))
    fix_thermr_geom = np.zeros(len(snapshots))

    for index in range(0,2):#len(snapshots)):
        tree_indexes, rays_T, rays_den, rays, radii, rays_vol = ray_maker(snapshots[index], m)
        rays_kappa, rays_cumulative_kappas, rays_photo, rays_index_photo = get_photosphere(rays_T, rays_den, radii)
        dim_ph = np.zeros(len(rays_index_photo))
        for j in range(len(rays_index_photo)):
            find_index_cell = int(rays_index_photo[j])
            vol_ph = rays_vol[j][find_index_cell]
            dim_ph[j] = (3 * vol_ph /(4 * np.pi))**(1/3)
            print('Simulation cell R: ' + str(dim_ph[j]))
            print('Our grid: ' + str(radii[find_index_cell+1]-radii[find_index_cell]))

        # with open('data/red/photosphere' + str(snapshots[index]) + '.txt', 'a') as fileph:
        #     fileph.write(' '.join(map(str,rays_photo)) + '\n')
        #     fileph.close()

        fix_photo_arit[index] = np.mean(rays_photo)
        fix_photo_geom[index] = gmean(rays_photo)

        tau, thermr, cumulative_taus = get_thermr(rays_T, rays_den, radii)
        fix_thermr_arit[index] = np.mean(thermr)
        fix_thermr_geom[index] = gmean(thermr)

        if plot_ph:
            fig, ax = plt.subplots(figsize = (6,8))
            img = ax.scatter(np.arange(192), rays_photo, c = dim_ph, s = 15)
            cbar = fig.colorbar(img)
            cbar.set_label(r'Cell dimension [$R_\odot$]')
            plt.axhline(np.mean(rays_photo), c = 'r', linestyle = '--', label = r'$\bar{R}_{ph}$ arit mean')
            plt.axhline(gmean(rays_photo), c = 'b', linestyle = '--', label = r'$\bar{R}_{ph}$ geom mean')
            plt.xlabel('Observers')
            plt.ylabel('$\log_{10} R_{ph} [R_\odot]$')
            plt.yscale('log')
            plt.grid()
            plt.legend()
            plt.show()   

        print('Snapshot ' + str(snapshots[index]))

    if plot_radii:
        with open('data/special_radii_m' + str(m) + '.txt', 'a') as file:
                file.write('# t/t_fb \n')
                file.write(' '.join(map(str, days)) + '\n')
                file.write('# Photosphere arithmetic mean \n')
                file.write(' '.join(map(str, fix_photo_arit)) + '\n')
                file.write('# Photosphere geometric mean \n')
                file.write(' '.join(map(str, fix_photo_geom)) + '\n')
                # file.write('# Thermalisation radius arithmetic mean \n')
                # file.write(' '.join(map(str, fix_thermr_arit)) + '\n')
                # file.write('# Thermalisation radius geometric mean \n')
                # file.write(' '.join(map(str, fix_thermr_geom)) + '\n')
                file.close()
        plt.plot(days, fix_photo_arit, '-o', color = 'black', label = 'Photosphere radius, arithmetic mean')
        plt.plot(days, fix_photo_geom, '-o', color = 'pink', label = 'Photosphere radius, geometric mean')
        # plt.plot(days, fix_thermr_arit, '-o', color = 'b', label = 'Thermalization radius, arithmetic mean')
        # plt.plot(days, fix_thermr_geom, '-o', color = 'r', label = 'Thermalization radius, geometric mean')
        plt.xlabel(r't/$t_{fb}$')
        plt.ylabel(r'Photosphere [$R_\odot$]')
        plt.grid()
        plt.legend()
    plt.show()

        