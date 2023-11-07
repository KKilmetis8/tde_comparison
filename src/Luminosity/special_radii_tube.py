#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 18:09:43 2023

@author: konstantinos
"""
import numpy as np
import numba
from src.Opacity.opacity_table import opacity
# Constants
c_cgs = 3e10 # [cm/s]
Rsol_to_cm = 6.957e10 # [cm]
def get_kappa(T: float, rho: float, dr: float):
    '''
    Calculates the integrand of eq.(8) Steinberg&Stone22.
    '''    
    # If there is nothing, the ray continues unimpeded
    if rho < np.exp(-49.3):
        return 0
    
    # Stream material, is opaque
    if T < np.exp(8.666): # 5802 K
        if rho > 1e-9:
            return 1 # testing 1
        else:
            return 0
    
    # Too hot: Kramers' law for absorption (planck)
    if T > np.exp(17.876): # 58_002_693
        X = 0.7389
        Z = 0.02
        kplanck =  1.2e26 * Z * (1 + X) * T**(-3.5) * rho #Kramers' bound-free opacity [cm^2/g]
        kplanck *= rho

        Tscatter = np.exp(17.87)
        kscattering = opacity(Tscatter, rho, 'scattering', ln = False)

        oppi = kplanck + kscattering
        tau_high = oppi * dr
        return tau_high 
    
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
    i = -1 # Initialize reverse loop
    while kappa <= threshold and i > -len(T):
        dr = rs[i]-rs[i-1] # Cell seperation
        new_kappa = get_kappa(T[i], rho[i], dr)
        kappa += new_kappa
        kappas.append(new_kappa)
        cumulative_kappas.append(kappa)
        i -= 1
    photo =  rs[i] #i it's negative
    return kappas, cumulative_kappas, photo

def get_photosphere(rays_T, rays_den, rays_R):
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
    rays_cumulative_kappas = []
    rays_photo = np.zeros(len(rays_T))
    
    for i in range(len(rays_T)):
        print('Ray maker ray: ', i) 
        # Isolate each ray
        T_of_single_ray = rays_T[i]
        Den_of_single_ray = rays_den[i]
        R_of_single_ray = rays_R[i]
        # Get photosphere
        _, cumulative_kappas, photo  = calc_photosphere(T_of_single_ray, 
                                                             Den_of_single_ray, 
                                                             R_of_single_ray)

        # Store
        # rays_kappas.append(kappas)
        rays_cumulative_kappas.append(cumulative_kappas)
        rays_photo[i] = photo

    return rays_cumulative_kappas, rays_photo