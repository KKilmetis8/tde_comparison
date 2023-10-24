#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 16:42:47 2023

@author: konstantinos

Equations refer to Krumholtz '07

NOTES FOR OTHERS:
- make changes in variables: m (power index of the BB mass), 
fixes (number of snapshots) anf thus days
"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

# Vanilla Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numba
import colorcet
# Custom Imports
from src.Calculators.ray_maker import ray_maker, find_observer
from src.Opacity.opacity_table import opacity
from src.Luminosity.photosphere import calc_photosphere, get_photosphere
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [5 , 4]


#%% Constants & Converter
c_cgs = 3e10 # [cm/s]
Rsol_to_cm = 6.957e10 # [cm]


#%%
##
# FUNCTIONS
##
###

def select_fix(m):
    if m == 4:
        snapshots = [233] #, 254, 263, 277 , 293, 308, 322]
        days = [1]# , 1.2, 1.3, 1.4, 1.56, 1.7, 1.8] 
    if m == 6:
        snapshots = [881]#[844, 881,  925, 950]
        days = [1.1]#[1, 1.1, 1.3, 1.4] # t/t_fb
    return snapshots, days

@numba.njit
def grad_calculator(ray, radii, sphere_radius): 
    sphere_radius_cgs = sphere_radius * Rsol_to_cm
    for i, radius in enumerate(radii):
        if radius > sphere_radius_cgs:
            idx = i - 1 
            break    
    step = radii[1] - radii[0]
    grad_E =  (ray[idx+1] - ray[idx]) / step 

    return grad_E, idx

def flux_calculator(grad_E, idx, 
                    rays, rays_T, rays_den):

    # Get opacity
    Energy = rays[idx]
    max_travel = c_cgs * Energy
    
    Temperature = rays_T[idx]
    Density = rays_den[idx]
    
    # Ensure we can interpolate
    rho_low = np.exp(-45)
    T_low = np.exp(8.77)
    T_high = np.exp(17.8)
    
    # If here is nothing, light continues
    if Density < rho_low:
        if (grad_E * max_travel) > 0:
            f = - max_travel
        else: 
            f = max_travel
    
    # Stream
    if Temperature < T_low:
        f = 0
    
    # T too high => Thompson opacity, we follow the table
    if Temperature > T_high:
        Temperature = np.exp(17.7)
        
    # Get Opacity, NOTE: Breaks Numba
    k_ross = opacity(Temperature, Density, 'rosseland', ln = False)
    
    # Calc R, eq. 28
    R = np.abs(grad_E) /  (k_ross * Energy)
    invR = 1 / R
    
    # Calc lambda, eq. 27
    coth = 1 / np.tanh(R)
    lamda = invR * (coth - invR)
    # Calc Flux, eq. 26
    Flux = - c_cgs * grad_E  * lamda / k_ross
    
    # Choose
    if Flux > max_travel:
        if (grad_E * max_travel) > 0:
            f = - max_travel
        else: 
            f = max_travel
    else:
        f = Flux
    return f
 
def doer_of_thing(fix, m):
    rays_T, rays_den, rays, radii, thetas, phis  = ray_maker(fix, m, select = True)
    _, _, photos = get_photosphere(rays_T, rays_den, radii)
    sphere_radius = np.mean(photos)/Rsol_to_cm
    new_rays_T, new_rays_den, new_rays, new_thetas, new_phis = find_observer(rays_T, rays_den, rays, thetas, phis, np.pi/2)
    
    # Calculate Flux
    grad_E, idx = grad_calculator(new_rays, radii, sphere_radius)
    flux = flux_calculator(grad_E, idx, 
                            new_rays, new_rays_T, new_rays_den)
    
    # Turn to luminosity
    sphere_radius *= Rsol_to_cm
    lum = flux * 4 * np.pi * sphere_radius**2
    print('Lum %.3e' % lum )
    return lum
#%%
##
# MAIN
##
if __name__ == "__main__":
    plot = True
    m = 6 # Choose BH
    fixes, days = select_fix(m)
    lums = []
            
    for fix in fixes:
        lum = doer_of_thing(fix, m)
        lums.append(lum)
    
    #%% Plotting
    if plot:
        plt.figure()
        plt.plot(days, lums, '-o', color = 'maroon')
        plt.yscale('log')
        plt.ylabel('Bolometric Luminosity [erg/s]')
        plt.xlabel('Days')
        if m == 6:
            plt.title('FLD for $10^6 \quad M_\odot$')
            plt.ylim(1e41,1e45)
        if m == 4:
            plt.title('FLD for $10^4 \quad M_\odot$')
            plt.ylim(1e39,1e42)
        plt.grid()
        plt.show()

