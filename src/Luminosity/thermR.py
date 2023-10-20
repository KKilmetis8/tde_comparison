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
        # print('T low')
        return 100
    
    # Too hot: Thompson Opacity.
    # Make it fall inside the table: from here the extrapolation is constant
    # This could be made faster
    if T > np.exp(17.876):
        # print('high T')
        T = np.exp(17.87)
    
    # Lookup table
    oppi = opacity(T, rho,'effective', ln = False)
    tau =  oppi * dr
    
    return tau

def calc_thermr(rs, T, rho, threshold = 1):
    '''
    Finds and saves the effective optical depth at every cell the ray passess through.
    We use it to find the thermr.

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
        
    thermr : float,
        Where the thermr is for that ray.
    '''
    tau = 0
    taus = []
    cumulative_taus = []
    dr = rs[1]-rs[0] # Cell seperation
    i = -1 # Initialize reverse loop
    #print('--new ray--')
    while tau < threshold and i > -len(T):
        new_tau = optical_depth(T[i], rho[i], dr)
        tau += new_tau
        taus.append(new_tau)
        cumulative_taus.append(tau)
        i -= 1
    thermr =  rs[i] #i it's negative
    return taus, thermr, cumulative_taus

def get_thermr(fix, m):
    ''' Wrapper function'''
    rays_T, rays_den, _, radii = ray_maker(fix, m)
    # Get the thermr
    rays_tau = []
    rays_cumulative_taus = []
    thermr = np.zeros(len(rays_T))
    
    for i in range(len(rays_T)):
        
        # Isolate each ray
        T_of_single_ray = rays_T[i]
        Den_of_single_ray = rays_den[i]
        
        # Get thermr
        taus, photo, cumulative_taus = calc_thermr(radii, T_of_single_ray, Den_of_single_ray, 
                                        threshold = 5)
        # Store
        rays_tau.append(taus)
        rays_cumulative_taus.append(cumulative_taus)
        thermr[i] = photo

    return rays_T, rays_den, rays_tau, thermr, radii, rays_cumulative_taus

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
        rays_T , rays_den , tau, photoo, radii, cumulative_taus = get_thermr(fix,m)
        photoo /=  6.957e10
    #%% Plot tau
    plot_tau = np.zeros( (len(radii), len(tau)))
    for i in range(192):
        for j in range(len(cumulative_taus)):
            temp = cumulative_taus[i][j]
            plot_tau[-j-1,i] =  temp
            if temp>5:
                plot_tau[0:-j, i ] = temp
                break
        plot_tau[0:-j, i ] = temp

    img = plt.pcolormesh(radii/6.957e10, np.arange(192), plot_tau.T, 
                          cmap = 'Greys', norm = colors.LogNorm(vmin = 1e-4, vmax =  5))
    cbar = plt.colorbar(img)
    plt.title('Rays')
    cbar.set_label('Optical depth')
    plt.xlabel('Distance from BH [$R_\odot$]')
    # plt.ylim(60,80)
    plt.ylabel('Observers')
    # plt.xscale('log')
    img.axes.get_yaxis().set_ticks([])

