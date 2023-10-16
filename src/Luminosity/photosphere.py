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
import colorcet
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [5 , 3]
plt.rcParams['axes.facecolor'] = 'whitesmoke'

# Custom Imports
from src.Opacity.opacity_table import opacity
from src.Calculators.ray_maker import ray_maker


################
# FUNCTIONS
################
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
        # T = np.exp(8.87)
        print('T low')
        return 1e4
    
    # Too hot: Thompson Opacity.
    # Make it fall inside the table: from here the extrapolation is constant
    # This could be made faster
    if T > np.exp(17.876):
        # print('high T')
        T = np.exp(17.7)
    
    # Lookup table
    oppi = opacity(T, rho,'effective', ln = False)
    tau =  oppi * dr
    
    return tau

def calc_photosphere(rs, T, rho, m, threshold = 1):
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
    print('--new ray--')
    while tau < threshold and i > -len(T):
        new_tau = optical_depth(T[i], rho[i], dr)
        tau += new_tau
        taus.append(new_tau)
        print('tau: ', tau)
        i -= 1

    photosphere =  rs[i] #i it's negative
    return taus, photosphere

def get_photosphere(fix, m, get_observer = False):
    ''' Wrapper function'''
    rays_T, rays_den, _, radii = ray_maker(fix, m)
    # Get the photosphere
    rays_tau = []
    photosphere = np.zeros(len(rays_T))
    
    for i in range(len(rays_T)):
        
        # Isolate each ray
        T_of_single_ray = rays_T[i]
        Den_of_single_ray = rays_den[i]
        
        # Get Photosphere
        taus, photo = calc_photosphere(radii, T_of_single_ray, Den_of_single_ray, 
                                       m, threshold = 5)
        # Store
        rays_tau.append(taus)
        photosphere[i] = photo

    return rays_T, rays_den, rays_tau, photosphere, radii

################
# MAIN
################

if __name__ == "__main__":
    m = 6 # M_bh = 10^m M_sol | Choose 4 or 6
    
    # Make Paths
    if m == 4:
        fixes = [233] #[233, 254, 263, 277, 293, 308, 322]
        loadpath = '4/'
    if m == 6:
        fixes = [844] #[844, 881, 925, 950]
        loadpath = '6/'

    for fix in fixes:
        rays_T , rays_den , tau, photoo, radii = get_photosphere(fix,m)
        photoo /=  6.957e10
    #%% Plot tau
    plot_tau = np.zeros( (len(radii), len(tau)))
    for i in range(192):
        for j in range(1000):
            temp = tau[i][j]
            plot_tau[-j,i] =  temp
            if temp>0:
                plot_tau[:-j, i ] = temp
                break

    img = plt.pcolormesh(radii/6.957e10, np.arange(len(tau)), plot_tau.T, 
                          cmap = 'Greys', vmin = 0, vmax =  5)
    cbar = plt.colorbar(img)
    plt.title('Rays')
    cbar.set_label('Optical depth')
    plt.xlabel('Distance from BH [$R_\odot$]')
    plt.ylabel('Observers')
    img.axes.get_yaxis().set_ticks([])