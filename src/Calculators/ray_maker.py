#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 10:19:34 2023

@author: konstantinos
"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

import numpy as np
import numba
import healpy as hp
import colorcet
from astropy.coordinates import cartesian_to_spherical
from src.Calculators.legion_of_casters import THROUPLE_S_CASTERS
import matplotlib.pyplot as plt
import matplotlib.colors as colors

#%% Constants & Converter
NSIDE = 4
G = 6.6743e-11 # SI
Msol = 1.98847e30 # kg
Rsol = 6.957e8 # m
t = np.sqrt(Rsol**3 / (Msol*G )) # Follows from G = 1
c = 3e8 * t/Rsol # simulator units. Need these for the PW potential
c_cgs = 3e10 # [cm/s]
Msol_to_g = 1.989e33 # [g]
Rsol_to_cm = 6.957e10 # [cm]
den_converter = Msol_to_g / Rsol_to_cm**3
en_den_converter = Msol_to_g / (Rsol_to_cm  * t**2 ) # Energy Density converter

def ray_maker(fix, m):
    fix = str(fix)
    Mbh = 10**m 
    Rt =  Mbh**(1/3) # Msol = 1, Rsol = 1
    
    # Import
    X = np.load( str(m) + '/'  + fix + '/CMx_' + fix + '.npy')
    Y = np.load( str(m) + '/'  + fix + '/CMy_' + fix + '.npy')
    Z = np.load( str(m) + '/'  + fix + '/CMz_' + fix + '.npy')
    Mass = np.load( str(m) + '/'  + fix + '/Mass_' + fix + '.npy')
    T = np.load( str(m) + '/'  + fix + '/T_' + fix + '.npy')
    Den = np.load( str(m) + '/'  + fix + '/Den_' + fix + '.npy')
    Rad = np.load( str(m) + '/'  +fix + '/Rad_' + fix + '.npy')

    # Convert Energy / Mass to Energy Density in CGS
    Rad *= Den 
    Rad *= en_den_converter
    Den *= den_converter 
    # Convert to spherical
    R, THETA, PHI = cartesian_to_spherical(X,Y,Z)
    R = R.value 
    THETA = THETA.value
    PHI = PHI.value
    
    #print('Den (weight) from simulation:', np.divide(np.sum(Den*Mass), np.sum(Mass)))

    # Ensure that the regular grid cells are smaller than simulation cells
    start = 100
    stop = 40 * Rt
    if m == 6:
        num = 600 + 1 # about the average of cell radius
    if m == 4:
        num = 500 #350
    radii = np.linspace(start, stop, num) #simulator units
    
    # Find observers with Healpix
    thetas = np.zeros(192)
    phis = np.zeros(192)
    observers = []
    for i in range(0,192):
        thetas[i], phis[i] = hp.pix2ang(NSIDE, i)
        thetas[i] -= np.pi/2 # Enforce theta in -pi to pi
        
        observers.append( (thetas[i], phis[i]) )
    
    #%% Cast
    T_casted, Den_casted, Rad_casted, counter = THROUPLE_S_CASTERS(radii, R, 
                                                       observers, THETA, PHI,
                                                       T, Den, Rad,
                                                       weights = Mass, 
                                                       avg = False)

    # Clean
    T_casted = np.nan_to_num(T_casted, neginf = 0)
    Den_casted = np.nan_to_num(Den_casted, neginf = 0)
    Rad_casted = np.nan_to_num(Rad_casted, neginf = 0)
    
    # DROP THE LAST ONE
    T_casted = np.delete(T_casted, -1, axis = 0)
    Den_casted = np.delete(Den_casted, -1, axis = 0)
    Rad_casted = np.delete(Rad_casted, -1, axis = 0)

    # Make Rays
    rays = []
    rays_den = []
    rays_T = []
    for i, observer in enumerate(observers):
        # Ray holds Erad
        rays.append(Rad_casted[: , i])

        # The Density in each ray
        d_ray = Den_casted[:, i]
        rays_den.append(d_ray)    

        # The Temperature in each ray
        t_ray = T_casted[:, i]
        rays_T.append(t_ray)
    
    radii *= Rsol_to_cm
    return rays_T, rays_den, rays, radii



if __name__ == '__main__':
   _, _, _, _ =  ray_maker(844, 6)