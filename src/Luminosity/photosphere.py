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
def get_kappa(T, rho, dr):
    '''
    Calculates the integrand from eq.(8) Steinber&Stone22.
    CHECK THE IFs
    '''    
    # If there is nothing, the ray continues unimpeded
    # if rho < np.exp(-49.3):
    if rho < np.exp(-23):
        return 0
    
    # Stream material, is opaque
    if T < np.exp(8.666):
        return 100
    
    # Too hot: Thompson Opacity.
    # Make it fall inside the table: from here the extrapolation is constant
    if T > np.exp(17.876):
        T = np.exp(17.87)
    
    # Lookup table
    kp = opacity(T, rho,'planck', ln = False)
    ks = opacity(T, rho,'scattering', ln = False)
    kappa =  (kp + ks) * dr
    
    return kappa

def calc_photosphere(rs, T, rho):
    '''
    Finds and saves the photosphere
    '''
    threshold = 2/3
    kappa = 0
    kappas = []
    dr = rs[1]-rs[0] # Cell seperation
    i = -1 # Initialize reverse loop
    while kappa < threshold and i > -len(T):
        new_kappa = get_kappa(T[i], rho[i], dr)
        kappa += new_kappa
        kappas.append(new_kappa)
        #print('kappa: ', kappa)
        i -= 1
    if kappa < threshold:
        print('Photoshpere not reached, you are at ', kappa)

    photo =  rs[i] #i it's negative
    return kappas,photo

def get_photosphere(fix, m):
    ''' Wrapper function'''
    rays_T, rays_den, _, radii = ray_maker(fix, m)
    # Get the thermr
    rays_kappa = []
    photos = np.zeros(len(rays_T))
    
    for i in range(len(rays_T)):
        # Isolate each ray
        T_of_single_ray = rays_T[i]
        Den_of_single_ray = rays_den[i]
        
        # Get photosphere
        kappas, photo = calc_photosphere(radii, T_of_single_ray, Den_of_single_ray)
        # Store
        rays_kappa.append(kappas)
        photos[i] = photo

    return rays_T, rays_den, rays_kappa, photos, radii

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
        rays_T, rays_den, rays_kappa, photos, radii = get_photosphere(fix,m)
        photos /=  6.957e10
    
    plot_kappa = np.zeros( (len(radii), len(rays_kappa)))
    for i in range(192):
        for j in range(len(rays_kappa[i])):
            temp = rays_kappa[i][j]
            plot_kappa[-j-1,i] =  temp
            if temp > 2/3:
                plot_kappa[0:-j, i ] = temp
                break

    img = plt.pcolormesh(radii/6.957e10, np.arange(192), plot_kappa.T, 
                          cmap = 'Oranges', norm = colors.LogNorm(vmin = 1e-6, vmax =  2/3))
    cbar = plt.colorbar(img)
    plt.title('Rays')
    cbar.set_label('Photosphere')
    plt.xlabel('Distance from BH [$R_\odot$]')
    plt.ylabel('Observers')
    # plt.xscale('log')
    img.axes.get_yaxis().set_ticks([])
    plt.xlim(0,1500)
    plt.savefig('Figs/photosphere.png')
    plt.show()
