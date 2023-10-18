#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 12:29:28 2023

@author: konstantinos
"""

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

# Vanilla Imports
import numpy as np
import numba
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import colorcet
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [5 , 3]
plt.rcParams['axes.facecolor'] = 'whitesmoke'

# Custom Imports
from src.Opacity.opacity_table import opacity
from src.Calculators.ray_maker import ray_maker
Rsol_to_cm = 6.957e10 # [cm]


################
# FUNCTIONS
################
def optical_depth(T, rho):
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
        # T = np.exp(8.87)
        # print('T low')
        return 100
    
    # Too hot: Thompson Opacity.
    # Make it fall inside the table: from here the extrapolation is constant
    if T > np.exp(17.876):
        # print('high T')
        T = np.exp(17.87)
        # X = 0.734
        # thompson = rho * 0.2 * (1 + X) # 1/cm units
        # print('Thompson: ', thompson)
        # print('Table: ', opacity(T, rho,'effective', ln = False))
        # return thompson
    
    # Lookup table
    # print('rho: ', np.log(rho))
    tau = opacity(T, rho,'effective', ln = False)
    
    return tau

def calc_photosphere(rs, T, rho, threshold = 1):
    '''
    Finds and saves the effective optical depth at every cell the ray passess through.
    We use it to find the photosphere.

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
        The optical depth of every cell.
        
    photosphere : float,
        Where the photosphere is for that ray.
    '''
    tau = 0
    taus = []
    dr = rs[1]-rs[0] # Cell seperation
    i = -1 # Initialize reverse loop
    while tau < threshold and i > -len(T):
        new_tau = optical_depth(T[i], rho[i]) * dr
        tau += new_tau
        taus.append(tau)
        i -= 1

    photosphere =  rs[i] #i it's negative
    return taus, photosphere
#%%
################
# MAIN
################

m = 6 # M_bh = 10^m M_sol | Choose 4 or 6

# Make Paths
if m == 4:
    fixes = [233] #[233, 254, 263, 277, 293, 308, 322]
    loadpath = '4/'
if m == 6:
    fixes = [844] #[844, 881, 925, 950]
    loadpath = '6/'

for fix in fixes:
    rays_T, rays_den, _, radii = ray_maker(fix, m)
    # Get the photosphere

    #%%
    rays_tau = []
    photosphere = np.zeros(len(rays_T))
    for i in range(len(rays_T)):
        
        # Isolate each ray
        T_of_single_ray = rays_T[i]
        Den_of_single_ray = rays_den[i]
        
        # Get Photosphere
        taus, photo = calc_photosphere(radii, T_of_single_ray, Den_of_single_ray, 
                                       threshold = 5)
        # Store
        rays_tau.append(taus)
        photosphere[i] = photo

    photosphere /=  6.957e10
    # Plot tau
    plot_tau = np.zeros( (len(radii), 192))
    for i in range(192):
        for j in range(len(rays_tau[i])):
            temp = rays_tau[i][j]
            j -= 1
            plot_tau[-j,i] =  temp
            if temp > 5:
                plot_tau[0:-j,i] = temp
                break
    
    img = plt.pcolormesh(radii/6.957e10, np.arange(192), plot_tau.T, 
                          cmap = 'Greys', norm = colors.LogNorm(vmin = 1e-6, vmax =  5))
    cbar = plt.colorbar(img)
    plt.title('Rays')
    cbar.set_label('Optical depth')
    plt.xlabel('Distance from BH [$R_\odot$]')
    plt.ylabel('Observers')
    # plt.xscale('log')
    img.axes.get_yaxis().set_ticks([])