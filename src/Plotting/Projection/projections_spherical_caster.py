#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 09:48:30 2023

@author: konstantinos
"""

import sys
sys.path.append('/Users/paolamartire/tde_comparison')

import numpy as np
import numba
import healpy as hp
import colorcet
from astropy.coordinates import cartesian_to_spherical
from astropy.coordinates import spherical_to_cartesian
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
den_converter = Msol_to_g / Rsol_to_cm**2
en_den_converter = Msol_to_g / (Rsol_to_cm  * t**2 ) # Energy Density converter


def evoker(fix, m, select = False):
    ''' Evokes the caster '''
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
    

    # Ensure that the regular grid cells are smaller than simulation cells
    start = 2 * Rt
    stop = 10_000 #400 * Rt
    if m == 6:
        num = 750 + 1 # about the average of cell radius
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
    
    # Cast
    T_casted, Den_casted, Rad_casted = THROUPLE_S_CASTERS(radii, R, 
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
    radii = np.delete(radii, -1, axis = 0)
    radii *= Rsol_to_cm
    
    return T_casted, Den_casted, radii, thetas, phis
T_cast, Den_cast, radii, thetas, phis = evoker(881, 6)
#%% Transform to 2D

Den_cast = np.log10(Den_cast)
Den_cast = np.nan_to_num(Den_cast, neginf = 0)

fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
img = ax.pcolormesh(radii, phis, Den_cast.T, cmap='cet_fire')
plt.colorbar(img)
