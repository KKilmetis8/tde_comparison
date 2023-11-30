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

Rsol_to_cm = 6.957e10 # [cm]
################
# FUNCTIONS
################

def select_fix(m):
    if m == 4:
        snapshots = [233] #, 254, 263, 277 , 293, 308, 322]
        days = [1]# , 1.2, 1.3, 1.4, 1.56, 1.7, 1.8] 
    if m == 6:
        snapshots = [844, 881, 925, 950, 980, 1008] 
        days = [1, 1.1, 1.3, 1.4, 1.5, 1.6] 
    return snapshots, days

def get_kappa(T: float, rho: float, r_dlogr: float):
    '''
    Calculates the integrand of eq.(8) Steinberg&Stone22.

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
    kappar: int.
            The "optical depth" of a cell.
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
        tau_high = oppi * r_dlogr
        return tau_high 
    
    # Lookup table
    k = opacity(T, rho,'red', ln = False)
    kappar =  k * r_dlogr
    
    return kappar

def calc_photosphere(T, rho, radius, branch_indexes):
    '''
    Finds and saves the photosphere (CGS) for ONE every ray.

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
    photo: int
           Photosphere (CGS) 
    index_ph: int
              photosphere's index in our radius.
    branch_index_ph: int.
                    Photosphere index in the tree.
    '''
    threshold = 2/3
    kappa = 0
    kappas = []
    cumulative_kappas = []
    i = -1 # Initialize reverse loop
    while kappa <= threshold and i > -len(T):
        dlogr = np.log(radius[i]) - np.log(radius[i-1]) #back to logspace
        r_dlogr = radius[i] * dlogr #to integrate in log space
        new_kappa = get_kappa(T[i], rho[i], r_dlogr)
        kappa += new_kappa
        kappas.append(new_kappa)
        cumulative_kappas.append(kappa)
        i -= 1

    photo =  radius[i] #i it's negative
    index_ph = i + len(T) 
    branch_index_ph = branch_indexes[i]

    return kappas, cumulative_kappas, photo, index_ph, branch_index_ph

def get_photosphere(rays_T, rays_den, radius, tree_indexes):
    '''
    Finds and saves the photosphere (CGS) at every ray.

    Parameters
    ----------
    rays_T: n-D arrays.
            Temperature of every ray/cell (CGS).
    rays_den: n-D arrays.
            Density of every ray/cell (CGS).
    radius: 1D array.
            Radius (CGS).
    tree_indexes: 1D array.
                Tree indexes for cells in the rays.

    Returns
    -------
    rays_kappas: nD array.
                The "optical depth" of a single cell in every ray. 
    rays_cumulative_kappas: nD array.
                The total "optical depth" of a single cell in every ray.
    ray_photo: 1D array.
                Photosphere value (CGS).
    rays_index_photo: 1D array.
                        Photosphere index in our radius.
    tree_index_photo: 1D array.
                    Photosphere index in the tree.
    '''
    rays_kappas = []
    rays_cumulative_kappas = []
    rays_photo = np.zeros(len(rays_T))
    rays_index_photo = np.zeros(len(rays_T))
    tree_index_photo = np.zeros(len(rays_T))
    
    for i in range(len(rays_T)):
        # Isolate each ray
        T_of_single_ray = rays_T[i]
        Den_of_single_ray = rays_den[i]
        branch_indexes = tree_indexes[i]
        
        # Get photosphere
        kappas, cumulative_kappas, photo, index_ph, branch_index_ph  = calc_photosphere(T_of_single_ray, Den_of_single_ray, 
                                                                       radius, branch_indexes)

        # Store
        rays_kappas.append(kappas)
        rays_cumulative_kappas.append(cumulative_kappas)
        rays_photo[i] = photo
        rays_index_photo[i] = index_ph
        tree_index_photo[i] = branch_index_ph

    return rays_kappas, rays_cumulative_kappas, rays_photo, rays_index_photo, tree_index_photo


def optical_depth(T: float, rho: float, r_dlogr: float):
    '''
    Calculates the optical depth at a point

    Parameters
    ----------
    T : float,
        Temperature in [cgs]. 
    rho : float. 
        Density in [cgs]. 

    r_dlogr : float,
              Delta integration in log space in [cgs].

    Returns
    -------
    tau : float,
        The optical depth in [cgs].
    '''    
    # If there is nothing, the ray continues unimpeded
    if rho < np.exp(-49.3):
        return 0
    
    # Stream material, is opaque
    if T < np.exp(8.666):
        return 100
    
    # Too hot: Kramers' law
    if T > np.exp(17.876):
        X = 0.7389
        Z = 0.02
        kplanck = 3.68e22 * (1-Z) * (1 + X) * T**(-3.5) * rho #Kramers' free-free opacity [cm^2/g]
        kplanck *= rho
        
        Tscatter = np.exp(17.87)
        kscattering = opacity(Tscatter, rho,'scattering', ln = False)
        oppi = np.sqrt(3 * kplanck * (kplanck + kscattering)) 
        tau_high = oppi * r_dlogr
        
        return tau_high 
    
    # Lookup table
    oppi = opacity(T, rho,'effective', ln = False)
    tau =  oppi * r_dlogr
    
    return tau

def calc_thermr(T, rho, radius, branch_indexes, threshold = 5):
    '''
    Finds and saves the effective optical depth at every cell the ray passess through.
    We use it to find the thermr of the ray.

    Parameters
    ----------
    radius : np.array
        Radial coordinates of a ray in [cgs].
    rho : np.array,
        Densities in a ray in [cgs].
    T : np.array,
        Temperatures in a ray
    branch_indexes: np.array
                    tree indexes of the cell stored in our radius
    threshold : float, optional
        The desired optical depth in [cgs].

    Returns
    -------
    taus : np.array,
        The optical depth of a single cell.
        
    cumulative_taus : np.array,
        The total optical depth of a single cell.

    thermr : float,
        Where the thermr is for that ray.
    '''
    tau = 0
    taus = []
    cumulative_taus = []
    i = -1 # Initialize reverse loop
    while tau < threshold and i > -len(T):
        dlogr = np.log(radius[i]) - np.log(radius[i-1]) #back to logspace
        r_dlogr = radius[i] * dlogr #to integrate in log space
        new_tau = optical_depth(T[i], rho[i], r_dlogr)
        tau += new_tau
        taus.append(new_tau)
        cumulative_taus.append(tau)
        i -= 1
    thermr =  radius[i] #i it's negative
    index_term = branch_indexes[i]

    return taus, cumulative_taus, thermr, index_term

def get_thermr(rays_T, rays_den, radii, tree_indexes):
    '''
    Finds and saves the thermalisation radius (CGS) for every ray.

    Parameters
    ----------
    rays_T, rays_den: n-D arrays (CGS)
    radii: 1D array (CGS)

    Returns
    -------
    rays_taus, rays_cumulative_taus: nD arrays.
    ray_thermr: 1D array.
    rays_index_thermr: 1D array.
    '''
    rays_tau = []
    rays_cumulative_taus = []
    rays_thermr = np.zeros(len(rays_T))
    rays_index_therm = np.zeros(len(rays_T))
    
    for i in range(len(rays_T)):
        # Isolate each ray
        T_of_single_ray = rays_T[i]
        Den_of_single_ray = rays_den[i]
        branch_indexes = tree_indexes[i]
        
        # Get thermr
        taus, cumulative_taus, thermr, index_term = calc_thermr(T_of_single_ray, Den_of_single_ray, 
                                                                radii, branch_indexes)
        # Store
        rays_tau.append(taus)
        rays_cumulative_taus.append(cumulative_taus)
        rays_thermr[i] = thermr
        rays_index_therm[i] = int(index_term)

    return rays_tau, rays_cumulative_taus, rays_thermr, rays_index_therm


################
# MAIN
################

if __name__ == "__main__":
    photosphere = True
    thermalisation = True
    
    plot = False
    m = 6 
    loadpath = str(m) + '/'
    snapshots, days = select_fix(m)

    fix_photo_arit = np.zeros(len(snapshots))
    fix_photo_geom = np.zeros(len(snapshots))
    fix_thermr_arit = np.zeros(len(snapshots))
    fix_thermr_geom = np.zeros(len(snapshots))

    for index in range(0,4):#len(snapshots)):
        print('Snapshot ' + str(snapshots[index]))
        tree_indexes, rays_T, rays_den, rays, radii, rays_vol = ray_maker(snapshots[index], m)

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

        # with open('data/red/photosphere' + str(snapshots[index]) + '.txt', 'a') as fileph:
        #     fileph.write(' '.join(map(str,rays_photo)) + '\n')
        #     fileph.close()

        if photosphere:
            rays_kappa, rays_cumulative_kappas, rays_photo, _, _ = get_photosphere(rays_T, rays_den, radii, tree_indexes)
            rays_photo = rays_photo/Rsol_to_cm

            fix_photo_arit[index] = np.mean(rays_photo)
            fix_photo_geom[index] = gmean(rays_photo)

            if plot:
                fig, ax = plt.subplots(figsize = (8,6))
                img = ax.scatter(np.arange(192), rays_photo, c = 'k', s = 15)
                cbar = fig.colorbar(img)
                cbar.set_label(r'Cell dimension [$R_\odot$]')
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
            rays_tau, rays_cumulative_taus, rays_thermr, rays_index_therm = get_thermr(rays_T, rays_den, radii, tree_indexes)
            rays_thermr = rays_thermr/Rsol_to_cm

            fix_thermr_arit[index] = np.mean(rays_thermr)
            fix_thermr_geom[index] = gmean(rays_thermr)
            
            if plot: 
                fig, ax = plt.subplots(figsize = (8,6))
                img = ax.scatter(np.arange(192), rays_thermr, c = 'k', s = 15)
                cbar = fig.colorbar(img)
                cbar.set_label(r'Cell dimension [$R_\odot$]')
                plt.axhline(np.mean(rays_thermr), c = 'r', linestyle = '--', label = r'$\bar{R}_{ph}$ arit mean')
                plt.axhline(gmean(rays_thermr), c = 'b', linestyle = '--', label = r'$\bar{R}_{ph}$ geom mean')
                plt.axhline(800, c = 'r', label = r'$\bar{R}_{ph}$ arit mean') #Elad
                plt.axhline(50, c = 'b', label = r'$\bar{R}_{ph}$ geom mean') #Elad
                plt.xlabel('Observers')
                plt.ylabel('$\log_{10} R_{ph} [R_\odot]$')
                plt.yscale('log')
                plt.grid()
                plt.legend()
                plt.savefig('therm_obs' + str(snapshots[index]) + '.png')
                plt.show()   


    if photosphere:         
        with open('data/special_radii_m' + str(m) + '.txt', 'a') as file:
            file.write('# t/t_fb \n')
            file.write(' '.join(map(str, days)) + '\n')
            file.write('# Photosphere arithmetic mean \n')
            file.write(' '.join(map(str, fix_photo_arit)) + '\n')
            file.write('# Photosphere geometric mean \n')
            file.write(' '.join(map(str, fix_photo_geom)) + '\n')
            file.close()
            
    if thermalisation:
        with open('data/special_radii_m' + str(m) + '.txt', 'a') as file:
            file.write('# t/t_fb \n')
            file.write(' '.join(map(str, days)) + '\n')
            file.write('# Thermalisation radius arithmetic mean \n')
            file.write(' '.join(map(str, fix_thermr_arit)) + '\n')
            file.write('# Thermalisation radius geometric mean \n')
            file.write(' '.join(map(str, fix_thermr_geom)) + '\n')
            file.close()



            