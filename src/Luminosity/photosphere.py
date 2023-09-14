#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 16:02:41 2023

@author: konstantinos, paola

NOTES FOR OTHERS:
- things from snapshots are in solar and code units (mass in M_sol, lenght in R_sol, time s.t. G=1), we have to convert them in CGS 
- change m, fixes, loadpath
"""

"""Vanilla Imports"""
import numpy as np
import numba
import healpy as hp

"""Custom Imports"""
from src.Calculators.romberg import romberg
from src.Optical_Depth.opacity_table import opacity
from astropy.coordinates import cartesian_to_spherical
from src.Calculators.casters import THE_TRIPLE_CASTER

###
##
# VARIABLES
##
###

m = 4 #power index of the mass_BH: choose 4 or 6
"""Make Paths"""
if m == 4:
    fixes = np.arange(232,263 + 1)
    loadpath = '4/'
if m == 6:
    fixes = [844, 881, 925, 950]
    loadpath = '6/'

Mbh = 10**m # * Msol
Rt =  Mbh**(1/3) # Tidal radius (Msol = 1, Rsol = 1)
Msol_to_g = 1.989e33
Rsol_to_cm = 6.957e10
den_converter = Msol_to_g / Rsol_to_cm**3

###
##
# FUNCTIONS
##
###

def opt_depth(z, r, T):
    """Optical depth"""
    Tz = T((z,r))
    dr = np.abs(r[0]-r[1])
    tau = opacity(rz, Tz, 'effective', ln = False) * dr #CHANGE IT
    return tau

###
##
# MAIN
##
###

for fix in fixes:
    X = np.load( loadpath + fix + '/CMx_' + fix + '.npy')
    Y = np.load( loadpath + fix + '/CMy_' + fix + '.npy')
    Z = np.load( loadpath + fix + '/CMz_' + fix + '.npy')
    Mass = np.load( loadpath + fix + '/Mass_' + fix + '.npy')
    Den = np.load( loadpath + fix + '/Den_' + fix + '.npy')
    T = np.load( loadpath + '/T_' + fix + '.npy')
    
    Den *= den_converter # Convert to cgs
    R, THETA, PHI = cartesian_to_spherical(X,Y,Z) # Convert to Spherical
    R = R.value 
    THETA = THETA.value
    PHI = PHI.value
    
    """Ensure that the regular grid cells are smaller than simulation cell"""
    start = Rt
    stop = 500 * Rt
    if m == 6:
        num = 500 # about the average of cell radius
    if m == 4:
        num = 350
    radii = np.linspace(start, stop, num = 500)
    
    """Generate uniform observers"""
    NSIDE = 4 # 192 observers
    thetas = np.zeros(192)
    phis = np.zeros(192)
    for i in range(0,192):
       thetas[i], phis[i] = hp.pix2ang(NSIDE, i)
       thetas[i] -= np.pi/2
       phis[i] -= np.pi
       
    """Evoke!"""
    Den_casted = THE_TRIPLE_CASTER(radii, R, thetas, THETA, phis, PHI,
                      Den,
                      weights = Mass, avg = True) 
    T_casted = THE_TRIPLE_CASTER(radii, R, thetas, THETA, phis, PHI,
                      T, 
                      weights = Mass, avg = True)
    
    """Make into rays"""
    rays_den = []
    rays_T = []
    for i, theta in enumerate(thetas):
        for j, phi in enumerate(phis):
            
            # The Density in each ray
            d_ray = Den_casted[:, i , j]
            d_ray = np.nan_to_num(d_ray, neginf = 0)
            rays_den.append(d_ray)
            
            # The Temperature in each ray
            t_ray = T_casted[:, i , j]
            t_ray = np.nan_to_num(t_ray, neginf = 0)
            rays_T.append(t_ray)
            
    # Interpolate
    
    # Get Optical Depth
    # Keep integrating until it is one
    # Maybe use an equation solver for that?
    
    # 
# Stuff I copied from previous work that is bad and no one should witness 


opt4[i] = romberg(0.2 * Rt4 ,max_z4, opt_depth, r,
                      d4_inter, t4_inter) # Rsol / cm 