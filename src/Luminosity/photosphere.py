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
# from src.Calculators.romberg import romberg
from src.Optical_Depth.opacity_table import opacity
from astropy.coordinates import cartesian_to_spherical
from src.Calculators.spherical_caster import THE_SPHERICAL_CASTER

################
# CONSTANTS
################

Msol_to_g = 1.989e33
Rsol_to_cm = 6.957e10
den_converter = Msol_to_g / Rsol_to_cm**3

################
# FUNCTIONS
################
def optical_depth(rho, T, dr):
    '''
    Calculates the optical depth at a point.

    Parameters
    ----------
    rho : float,
        Density in [cgs].
    T : float,
        Temperature in [cgs].
    dr : float,
        Cell Size in R_sol.

    Returns
    -------
    tau : float,
        The optical depth in [cgs].
    '''
    
    # If the ray crosses the stream, it is opaque! τ = 1
    logT = np.log(T)
    logT = np.nan_to_num(logT, nan = 0, posinf = 0, neginf= 0)
    logrho = np.log(rho)
    logrho = np.nan_to_num(logrho, nan = 0, posinf = 0, neginf= 0)
    
    # If there is nothing, the ray continues unimpeded
    if logrho < -22 or logT < 1:
        return 0
    
    # Stream material, is opaque 
    if logT < 8.666:
        return 1
    
    # Convert Cell size to cgs
    dr *= Rsol_to_cm
    # Call opacity
    tau = opacity(logrho, logT, 'effective', ln = True) * dr 
    
    return tau

def calc_photosphere(rs, rho, T, threshold = 1):
    '''
    Finds the photosphere and saves the effective optical depth at every
    place the ray passess through.

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
    tau : float,
        The optical depth.
        
    photosphere : float,
        Where the photosphere is for that ray.
    '''
    tau = 0
    taus = []
    dr = rs[1]-rs[0] # Cell seperation
    i = -1 # Initialize reverse loop
    while tau < threshold and i > -350:
        new_tau = optical_depth(T[i], rho[i], dr)
        tau += new_tau
        taus.append(new_tau) 
        i -= 1
        
    photosphere =  rs[i]
    return taus, photosphere

################
# MAIN
################
def get_photosphere(fix, m):
    ''' Wrapper function'''
    Mbh = 10**m # * Msol
    Rt =  Mbh**(1/3) # Tidal radius (Msol = 1, Rsol = 1)
    
    fix = str(fix)
    loadpath = str(m) + '/'
    X = np.load( loadpath + fix + '/CMx_' + fix + '.npy')
    Y = np.load( loadpath + fix + '/CMy_' + fix + '.npy')
    Z = np.load( loadpath + fix + '/CMz_' + fix + '.npy')
    Mass = np.load( loadpath + fix + '/Mass_' + fix + '.npy')
    Den = np.load( loadpath + fix + '/Den_' + fix + '.npy')
    T = np.load( loadpath + fix + '/T_' + fix + '.npy')
    
    Den *= den_converter # Convert to cgs
    R, THETA, PHI = cartesian_to_spherical(X,Y,Z) # Convert to Spherical
    R = R.value 
    THETA = THETA.value
    PHI = PHI.value
    
    # Ensure that the regular grid cells are smaller than simulation cell
    start = Rt
    stop = 500 * Rt
    if m == 6:
        num = 500 # about the average of cell radius
    if m == 4:
        num = 350
    radii = np.linspace(start, stop, num)
    
    # Generate uniform observers"""
    NSIDE = 4 # 192 observers
    thetas = np.zeros(192)
    phis = np.zeros(192)
    observers = []
    for i in range(0,192):
       thetas[i], phis[i] = hp.pix2ang(NSIDE, i)
       thetas[i] -= np.pi/2
       phis[i] -= np.pi
       
       # Idea time
       observers.append( (thetas[i], phis[i]) )
       
    # Evoke!
    Den_casted = THE_SPHERICAL_CASTER(radii, R, observers, THETA, PHI,
                      Den,
                      weights = Mass, avg = True, loud = False) 
    T_casted = THE_SPHERICAL_CASTER(radii, R, observers, THETA, PHI,
                      T, 
                      weights = Mass, avg = True)
    Den_casted = np.nan_to_num(Den_casted, neginf = 0)
    T_casted = np.nan_to_num(T_casted, neginf = 0)
    
    plt.figure()
    plt.title('Den Casted')
    img = plt.imshow(Den_casted.T, cmap = 'cet_fire')
    cbar = plt.colorbar(img)
    #%% Make into rays OLD
    rays_den = []
    rays_T = []
    # for observer in observers:
    #     theta_obs = observer[0]
    #     phi_obs = observer[1]
        
    #     i = np.where(thetas == theta_obs)
    #     j = np.where(phis == phi_obs)
    #     print(i)
    #     # The Density in each ray
    #     d_ray = Den_casted[:, i , j]
    #     rays_den.append(d_ray)
        
    #     # The Temperature in each ray
    #     t_ray = T_casted[:, i , j]
    #     rays_T.append(t_ray)
        
    # Make into rays NEW
    for i, observer in enumerate(observers):
            
        # The Density in each ray
        d_ray = Den_casted[:, i]
        rays_den.append(d_ray)
        
        # The Temperature in each ray
        t_ray = T_casted[:, i]
        rays_T.append(t_ray)
        
    print('Shape Ray:',np.shape(rays_T))
    
    # img = plt.pcolormesh(radii, np.arange(len(rays_den)), rays_den, 
    #                      cmap = 'cet_fire')
    # cbar = plt.colorbar(img)
    # plt.title('Rays')
    # cbar.set_label('Radiation Energy Density')
    # plt.xlabel('r')
    # plt.ylabel('Various observers')
    # img.axes.get_yaxis().set_ticks([])
    
    # Get the photosphere
    rays_tau = []
    photosphere = np.zeros(len(rays_T))
    
    for i in range(len(rays_T)):
        
        
        # Isolate each ray
        T_of_single_ray = rays_T[i]
        Den_of_single_ray = rays_den[i]
        
        # Get Photosphere
        tau, photo = calc_photosphere(radii, Den_of_single_ray, 
                                      T_of_single_ray, threshold = 1)
        # Store
        rays_tau.append(tau)
        photosphere[i] = photo
    return rays_den, rays_T, rays_tau, photosphere, radii

if __name__ == "__main__":
    m = 6 # M_bh = 10^m M_sol | Choose 4 or 6
    
    # Make Paths
    if m == 4:
        fixes = 232 #np.arange(232,263 + 1)
        fix = 232
        loadpath = '4/'
    if m == 6:
        fixes = [844] #[844, 881, 925, 950]
        loadpath = '6/'

    for fix in fixes:
        get_photosphere(fix,m)