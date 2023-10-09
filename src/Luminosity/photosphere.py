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
from src.Optical_Depth.opacity_table import opacity
from astropy.coordinates import cartesian_to_spherical
from src.Calculators.spherical_caster import THE_SPHERICAL_CASTER

################
# CONSTANTS
################

Msol_to_g = 1.989e33
Rsol_to_cm = 6.957e10
den_converter = Msol_to_g / Rsol_to_cm**3
G = 6.6743e-11 # SI
Msol = 1.98847e30 # kg
Rsol = 6.957e8 # m
t = np.sqrt(Rsol**3 / (Msol*G )) # Follows from G = 1
num = 500

################
# FUNCTIONS
################
def optical_depth(T, rho, dr):
    '''
    Calculates the optical depth at a point.

    Parameters
    ----------
    T : float,
        Temperature in [cgs]. We will convert it in ln in order to interpolate.
    rho : float. 
        Density in [cgs]. We will convert it in ln in order to interpolate.

    dr : float,
        Cell Size in R_sol.

    Returns
    -------
    tau : float,
        The optical depth in [cgs].
    '''
    logT = np.log(T)
    logT = np.nan_to_num(logT, nan = 0, posinf = 0, neginf= 0) #posinf = big number?
    logrho = np.log(rho)
    logrho = np.nan_to_num(logrho, nan = 0, posinf = 0, neginf= 0) #posinf = big number?
    
    # If there is nothing, the ray continues unimpeded
    if logrho < -22 or logT < 1:
        return 0
    
    # Stream material, is opaque (???) CHECK CONDITION
    if logT < 8.666 or logT > 17.876:
        return 1
    
    # Convert Cell size to cgs
    dr *= Rsol_to_cm
    tau = opacity(logT, logrho,'effective', ln = True) * dr 
    
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
    while tau < threshold and i > -num:
        new_tau = optical_depth(T[i], rho[i], dr)
        tau += new_tau
        taus.append(new_tau) 
        i -= 1

    photosphere =  rs[i] #i it's negative
    return taus, photosphere

def get_photosphere(fix, m, red = False, get_observer = False):
    ''' Wrapper function'''
    Mbh = 10**m # * Msol
    Rt =  Mbh**(1/3) # Tidal radius in simulator units (Msol = 1, Rsol = 1)
    
    fix = str(fix)
    loadpath = str(m) + '/'
    X = np.load( loadpath + fix + '/CMx_' + fix + '.npy')
    Y = np.load( loadpath + fix + '/CMy_' + fix + '.npy')
    Z = np.load( loadpath + fix + '/CMz_' + fix + '.npy')
    Mass = np.load( loadpath + fix + '/Mass_' + fix + '.npy')
    Den = np.load( loadpath + fix + '/Den_' + fix + '.npy')
    T = np.load( loadpath + fix + '/T_' + fix + '.npy')
    # Convert to CGS and spherical units
    Den *= den_converter 
    R, THETA, PHI = cartesian_to_spherical(X,Y,Z) 
    R = R.value 
    THETA = THETA.value
    PHI = PHI.value
    
    # Ensure that the regular grid cells are smaller than simulation cell
    start = Rt
    stop = 500 * Rt
    radii = np.linspace(start, stop, num)
    
    # Generate uniform observers
    NSIDE = 4 # 192 observers
    thetas = np.zeros(192)
    phis = np.zeros(192)
    observers = []
    for i in range(0,192):
       thetas[i], phis[i] = hp.pix2ang(NSIDE, i)
       thetas[i] -= np.pi/2
       phis[i] -= np.pi
        
       observers.append( (thetas[i], phis[i]) )
       
    # Evoke!
    Den_casted = THE_SPHERICAL_CASTER(radii, R, observers, THETA, PHI,
                      Den,
                      weights = Mass, avg = True, loud = False) 
    T_casted = THE_SPHERICAL_CASTER(radii, R, observers, THETA, PHI,
                      T, 
                      weights = Mass, avg = True, loud = False)
    Den_casted = np.nan_to_num(Den_casted, neginf = 0)


    T_casted = np.nan_to_num(T_casted, neginf = 0) 
    # plt.figure()
    # plt.title('Den Casted')
    # img = plt.imshow(Den_casted.T, cmap = 'cet_fire')
    # cbar = plt.colorbar(img)
    #%% Make into rays 
    rays_den = []
    rays_T = []
        
    for i, observer in enumerate(observers):
            
        # The Density in each ray
        d_ray = Den_casted[:, i]
        rays_den.append(d_ray)
        
        # The Temperature in each ray
        t_ray = T_casted[:, i]
        rays_T.append(t_ray)

    if red:
        en_den_converter = Msol_to_g / (Rsol_to_cm  * t**2 ) # Energy Denstiy converter
        Rad = np.load( loadpath +fix + '/Rad_' + fix + '.npy')
        Rad *= Den 
        Rad *= en_den_converter 
        Rad_casted = THE_SPHERICAL_CASTER(radii, R, observers, THETA, PHI,
                        Rad,
                        weights = Mass, avg = True, loud = False)
        Rad_casted = np.nan_to_num(Rad_casted)
        rays = []
        for i, observer in enumerate(observers):
            # Ray holds Erad
            rays.append(Rad_casted[: , i])
    
    # Get the photosphere
    rays_tau = []
    photosphere = np.zeros(len(rays_T))
    
    for i in range(len(rays_T)):
        
        # Isolate each ray
        T_of_single_ray = rays_T[i]
        Den_of_single_ray = rays_den[i]
        
        # Get Photosphere
        taus, photo = calc_photosphere(radii, T_of_single_ray, Den_of_single_ray, 
                                       m, threshold = 1)
        # Store
        rays_tau.append(taus)
        photosphere[i] = photo

    # img = plt.pcolormesh(radii, np.arange(len(rays_tau)), rays_tau, 
    #                      cmap = 'cet_gouldian')
    # cbar = plt.colorbar(img)
    # plt.title('Rays')
    # cbar.set_label('Optical depth')
    # plt.xlabel('r')
    # plt.ylabel('Observers')
    # img.axes.get_yaxis().set_ticks([])
    if red:
        return rays_den, rays_T, rays, rays_tau, photosphere, radii
    else: 
        return rays_den, rays_T, rays_tau, photosphere, radii

################
# MAIN
################

if __name__ == "__main__":
    m = 4 # M_bh = 10^m M_sol | Choose 4 or 6
    
    # Make Paths
    if m == 4:
        fixes = [233] #[233, 254, 263, 277, 293, 308, 322]
        loadpath = '4/'
    if m == 6:
        fixes = [844] #[844, 881, 925, 950]
        loadpath = '6/'

    for fix in fixes:
        _, _ , tau, photoo, _ = get_photosphere(fix,m)