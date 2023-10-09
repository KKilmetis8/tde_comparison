#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 16:42:47 2023

@author: konstantinos

Equations refer to Krumholtz '07

NOTES FOR OTHERS:
- make changes in variables: m (power index of the BB mass), fixes (number of snapshots) anf thus days
"""

import numpy as np
import matplotlib.pyplot as plt
import colorcet
import numba
from astropy.coordinates import cartesian_to_spherical
from src.Calculators.spherical_caster import THE_SPHERICAL_CASTER
from src.Optical_Depth.opacity_table import opacity
from src.Luminosity.photosphere import get_photosphere
import healpy as hp
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [5 , 3]


#%% Constants & Converter
G = 6.6743e-11 # SI
Msol = 1.98847e30 # kg
Rsol = 6.957e8 # m
t = np.sqrt(Rsol**3 / (Msol*G )) # Follows from G = 1
c = 3e8 * t/Rsol # simulator units. Need these for the PW potential
c_cgs = 3e10 # [cm/s]
Msol_to_g = 1.989e33 #[g]
Rsol_to_cm = 6.957e10 #[cm]
den_converter = Msol_to_g / Rsol_to_cm**3
energy_converter =  Msol_to_g * Rsol_to_cm**2 / (t**2) # Energy / Mass to cgs converter
en_den_converter = Msol_to_g / (Rsol_to_cm  * t**2 ) # Energy Denstiy converter

###
##
# VARIABLES
##
###

NSIDE = 4 # 192 observers

def select_fix(m):
    if m == 4:
        snapshots = [263, 277] #[233, 254, 263, 277, 293, 308, 322]
        days = [1.3, 1.4] #[1, 1.2, 1.3, 1.4, 1.56, 1.7, 1.8] 
    if m == 6:
        snapshots = [925]#[844, 881, 925, 950, 1006]
        days = [1.4]#[1, 1.1, 1.3, 1.4, 1.6] #t/t_fb
    return snapshots, days

### OLD
# if m == 4:
#     fixes =  np.arange(233,263+1) #from 233 to 263
#     days = [1.015, 1.025, 1.0325, 1.0435, 1.0525, 1.06, 1.07, 1.08, 1.0875, 1.0975, 1.1075, 1.115, 1.125, 1.135, 1.1425, 1.1525, 1.1625, 1.17, 1.18, 1.19, 1.1975, 1.2075, 1.2175, 1.2275, 1.235, 1.245, 1.255, 1.2625, 1.2725, 1.2825, 1.29] #t/t_fb
# if m == 6:
#     fixes = [844] #[844, 881, 925, 950] # 1006]
#     days = [1.00325, 1.13975, 1.302, 1.39425] # 1.6] #t/t_fb
###

#%% Data Load
###
##
# FUNCTIONS
##
###

def doer_of_thing(fix, m):
    fix = str(fix)
    Mbh = 10**m 
    Rt =  Mbh**(1/3) # Msol = 1, Rsol = 1
    rg = 2*Mbh/c**2 #tidal radius in simulater units
    if m == 4:
        folder = '4/'
    else:
        folder = '6/'
        
    X = np.load( folder +fix + '/CMx_' + fix + '.npy')
    Y = np.load( folder +fix + '/CMy_' + fix + '.npy')
    Z = np.load( folder +fix + '/CMz_' + fix + '.npy')
    Mass = np.load( folder +fix + '/Mass_' + fix + '.npy')
    Den = np.load( folder +fix + '/Den_' + fix + '.npy')
    Rad = np.load( folder +fix + '/Rad_' + fix + '.npy')
    T = np.load( folder +fix + '/T_' + fix + '.npy')
    
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
    start = Rt
    stop = 500 * Rt
    if m ==6:
        num = 500 # about the average of cell radius
    if m == 4:
        num = 500 #350
    radii = np.linspace(start, stop, num)
    
    # Find observers with Healpix
    thetas = np.zeros(192)
    phis = np.zeros(192)
    observers = []
    for i in range(0,192):
        thetas[i], phis[i] = hp.pix2ang(NSIDE, i)
        thetas[i] -= np.pi/2
        phis[i] -= np.pi
        
        observers.append( (thetas[i], phis[i]) )
    
    #%% Cast
    Rad_casted = THE_SPHERICAL_CASTER(radii, R, observers, THETA, PHI,
                      Rad,
                      weights = Mass, avg = True, loud = False) 
    Den_casted = THE_SPHERICAL_CASTER(radii, R, observers, THETA, PHI,
                      Den,
                      weights = Mass, avg = True, loud = False) 
    T_casted = THE_SPHERICAL_CASTER(radii, R, observers, THETA, PHI,
                      T, 
                      weights = Mass, avg = True)
    Rad_casted = np.nan_to_num(Rad_casted)
    
    #%% Make Rays
    rays = []
    rays_den = []
    rays_T = []
    for i, observer in enumerate(observers):
        # Ray holds Erad
        rays.append(Rad_casted[: , i])

        # The Density in each ray
        d_ray = Den_casted[:, i]
        d_ray = np.log10(d_ray)
        d_ray = np.nan_to_num(d_ray, neginf = 0)
        rays_den.append(d_ray)    

        # The Temperature in each ray
        t_ray = T_casted[:, i]
        t_ray = np.log10(t_ray)
        t_ray = np.nan_to_num(t_ray, neginf = 0)
        rays_T.append(t_ray)
 
    #%% Let's see how it looks 
    print(np.shape(rays))
    img = plt.pcolormesh(radii, np.arange(len(rays)), rays, cmap = 'cet_gouldian')
    cbar = plt.colorbar(img)
    plt.title('Rays')
    cbar.set_label('Radiation Energy Density')
    plt.xlabel('r')
    plt.ylabel('Various observers')
    img.axes.get_yaxis().set_ticks([])
    plt.xscale('log')
    
    plt.figure()
    img = plt.pcolormesh(radii, np.arange(len(rays)), rays_den, cmap = 'cet_fire')
    cbar = plt.colorbar(img)
    plt.title('Rays')
    cbar.set_label(r'$log_{10}$Density')
    plt.xlabel('r')
    plt.ylabel('Various observers')
    img.axes.get_yaxis().set_ticks([])
    plt.xscale('log')
    
    plt.figure()
    img = plt.pcolormesh(radii, np.arange(len(rays)), rays_T, cmap = 'cet_bmy')
    cbar = plt.colorbar(img)
    plt.title('Rays')
    cbar.set_label('Temperature')
    plt.xlabel('r')
    plt.ylabel('Various observers')
    img.axes.get_yaxis().set_ticks([])
    plt.xscale('log')
    
    #%%
    
    def grad_calculator(rays, radii, sphere_radius = 15_000): 
        # Get the index of radius closest in sphere radius
        diffs = np.abs(radii - sphere_radius)
        idx = np.argmin(diffs)
        step = radii[idx + 1] - radii[idx]
        grad_E = np.zeros(len(rays))
        for i, ray in enumerate(rays):
            grad_E[i] = (ray[idx+1] - ray[idx]) / step 
        return grad_E, idx
    
    
    def flux_calculator(grad_E, idx, 
                     rays, rays_T, rays_den):
        print(rays_T)
        f = np.zeros(len(grad_E))
        max_count = 0
        zero_count = 0
        flux_count = 0
        for i, ray in enumerate(rays):
            # Get opacity
            Energy = ray[idx]
            max_travel = c_cgs * ray[idx]
            
            Temperature = rays_T[i][idx]
            Density = rays_den[i][idx]
            
            # If it is zero
            if Density == 0 or Temperature == 0:
                zero_count += 1
                f[i] = max_travel
                continue
            
            # Ensure that Density & Temperature are logs
            # NOTE: This is fucky and bad
            # To ensure I stay within interpolation range lnT>8.666
            # Low temperatures are assosciated with the stream, this is a zero
            # order way to discard the stream
            if Temperature < 8.666:
                continue 
                
            k_ross = opacity(Temperature, Density, 'rosseland', ln = True)
            
            # Calc R, eq. 28
            R = np.abs(grad_E[i]) /  (k_ross * Energy)
            invR = 1 / R
            
            # Calc lambda, eq. 27
            coth = 1 / np.tanh(R)
            lamda = invR * (coth - invR)
    
            # Calc Flux, eq. 26
            Flux =  max_travel * lamda / k_ross
            
            # Choose
            if Flux > max_travel:
                f[i] = max_travel
                max_count += 1
            else:
                f[i] = Flux
                flux_count += 1
                
        print('Max: ', max_count)
        print('Zero: ', zero_count)
        print('Flux: ', flux_count)
        return f
    
    if m == 6:
        sphere_radius = 35_000
    else:
        sphere_radius = 3_300 #7_000 for 277 and on

    # NEW: Find the correct spehere radius
    

    grad_E, radius_idx = grad_calculator(rays, radii, sphere_radius)
    flux = flux_calculator(grad_E, radius_idx, 
                           rays, rays_T, rays_den)
    
    # Divide by number of observers
    flux = np.sum(flux) / 192
    
    # Turn to luminosity
    sphere_radius *= Rsol_to_cm
    lum = flux * 4 * np.pi * sphere_radius**2
    print('%.7e' % lum )
    return lum

##
# MAIN
##
if __name__ == "__main__":
    lums = []
    m = 4 # Choose BH
    index = 0
    fixes, days = select_fix(m)
    for fix in fixes:
        lum = doer_of_thing(fix, m)
        lums.append(lum)
    
    #%%
    plt.figure()
    np.savetxt('try_reddata_m'+ str(m) + '.txt', (days, lums))
    plt.plot(days, lums, '-o', color = 'maroon')
    plt.yscale('log')
    plt.ylim(1e41,1e45)
    plt.ylabel('Bolometric Luminosity [erg/s]')
    plt.xlabel('Days')
    plt.title('FLD for $10^4 \quad M_\odot$')
    plt.grid()
    plt.show()

