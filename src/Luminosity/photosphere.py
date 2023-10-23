#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18

@author: Paola

Gives the photosphere for red
"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

# Vanilla Imports
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [5 , 3]
plt.rcParams['axes.facecolor'] = 'whitesmoke'

# Custom Imports
from src.Opacity.opacity_table import opacity

################
# FUNCTIONS
################
def get_kappa(T: float, rho: float, dr: float):
    '''
    Calculates the integrand of eq.(8) Steinberg&Stone22.
    CHECK THE IFs
    '''    
    # If there is nothing, the ray continues unimpeded
    if rho < np.exp(-49.3):
        return 0
    
    # Stream material, is opaque
    if T < np.exp(8.666):
        return 100
    
    # Too hot: Thompson Opacity.
    # Make it fall inside the table: from here the extrapolation is constant
    if T > np.exp(17.876):
        T = np.exp(17.87)
    
    # Lookup table
    k = opacity(T, rho,'red', ln = False)
    kappar =  k * dr
    
    return kappar

def calc_photosphere(T, rho, rs):
    '''
    Input: 1D arrays.
    Finds and saves the photosphere (in CGS).
    The kappas' arrays go from far to near the BH.
    '''
    threshold = 2/3
    kappa = 0
    kappas = []
    cumulative_kappas = []
    dr = rs[1]-rs[0] # Cell seperation
    i = -1 # Initialize reverse loop
    while kappa <= threshold and i > -len(T):
        new_kappa = get_kappa(T[i], rho[i], dr)
        kappa += new_kappa
        kappas.append(new_kappa)
        cumulative_kappas.append(kappa)
        i -= 1

    photo =  rs[i] #i it's negative
    return kappas, cumulative_kappas, photo

def get_photosphere(rays_T, rays_den, radii):
    '''
    Finds and saves the photosphere (in CGS) for every ray.

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
    
    for i in range(len(rays_T)):
        # Isolate each ray
        T_of_single_ray = rays_T[i]
        Den_of_single_ray = rays_den[i]
        
        # Get photosphere
        kappas, cumulative_kappas, photo  = calc_photosphere(T_of_single_ray, Den_of_single_ray, radii)

        # Store
        rays_kappas.append(kappas)
        rays_cumulative_kappas.append(cumulative_kappas)
        rays_photo[i] = photo

    return rays_kappas, rays_cumulative_kappas, rays_photo

################
# MAIN
################

if __name__ == "__main__":
    from src.Calculators.ray_maker import ray_maker

    m = 6 
    
    # Make Paths
    if m == 4:
        fixes = [233] #[233, 254, 263, 277, 293, 308, 322]
        loadpath = '4/'
    if m == 6:
        fixes = [844] #[844, 881, 925, 950]
        loadpath = '6/'

    for fix in fixes:
        rays_T, rays_den, _, radii = ray_maker(fix, m)
        rays_kappa, rays_cumulative_kappas, rays_photo = get_photosphere(rays_T, rays_den, radii)
        rays_photo /=  6.957e10
        print(np.max(rays_photo))
    
    plot_kappa = np.zeros( (len(radii), len(rays_kappa)))
    for i in range(192):
        for j in range(len(rays_cumulative_kappas)):
            temp = rays_cumulative_kappas[i][j]
            plot_kappa[-j-1,i] =  temp
            if temp > 2/3:
                print('Photosphere reached')
                plot_kappa[0:-j, i ] = temp
                break
        plot_kappa[0:-j, i ] = temp

    img = plt.pcolormesh(radii/6.957e10, np.arange(192), plot_kappa.T, 
                          cmap = 'Oranges', norm = colors.LogNorm(vmin = 1e-2, vmax =  2/3))
    cbar = plt.colorbar(img)
    plt.axvline(x=np.max(rays_photo), c = 'black', linestyle = '--')
    plt.title('Rays')
    cbar.set_label('Photosphere')
    plt.xlabel('Distance from BH [$R_\odot$]')
    plt.ylabel('Observers')
    img.axes.get_yaxis().set_ticks([])
    plt.savefig('Final plot/photosphere.png')
    plt.show()
