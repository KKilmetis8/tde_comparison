#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 16:02:41 2023

@author: konstantinos, Paola

NOTES FOR OTHERS:
- things from snapshots are in solar and code units (mass in M_sol, lenght in R_sol, time s.t. G=1), we have to convert them in CGS 
- change m, fixes, loadpath
"""

# Vanilla Imports
import numpy as np
import numba
import healpy as hp

# Custom Imports
from src.Calculators.romberg import romberg
from src.Optical_Depth.opacity_table import opacity
from astropy.coordinates import cartesian_to_spherical
from src.Calculators.casters import THE_TRIPLE_CASTER

################
# CONSTANTS
################

m = 4 # M_bh = 10^m M_sol | Choose 4 or 6
# Make Paths
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
    
    # Convert Cell size to cgs
    dr *= Rsol_to_cm
    
    # Call opacity
    tau = opacity(rho, T, 'effective', ln = False) * dr 
    
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
    while tau < threshold:
        new_tau = optical_depth(rho[i], T[i], dr)
        tau += new_tau
        taus.append(new_tau) 
        i -= 1
        
    photosphere =  rs[i]
    return taus, photosphere

################
# MAIN
################
def get_photosphere(fixes, m):
    ''' Wrapper function'''
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
        
        # Ensure that the regular grid cells are smaller than simulation cell
        start = Rt
        stop = 500 * Rt
        if m == 6:
            num = 500 # about the average of cell radius
        if m == 4:
            num = 350
        radii = np.linspace(start, stop, num = 500)
        
        # Generate uniform observers"""
        NSIDE = 4 # 192 observers
        thetas = np.zeros(192)
        phis = np.zeros(192)
        for i in range(0,192):
           thetas[i], phis[i] = hp.pix2ang(NSIDE, i)
           thetas[i] -= np.pi/2
           phis[i] -= np.pi
           
        # Evoke!
        Den_casted = THE_TRIPLE_CASTER(radii, R, thetas, THETA, phis, PHI,
                          Den,
                          weights = Mass, avg = True) 
        T_casted = THE_TRIPLE_CASTER(radii, R, thetas, THETA, phis, PHI,
                          T, 
                          weights = Mass, avg = True)
        
        # Make into rays
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
