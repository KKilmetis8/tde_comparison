#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 16:42:47 2023

@author: konstantinos

Equations refer to Krumholtz '07
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [5 , 3]
import colorcet
import numba
from astropy.coordinates import cartesian_to_spherical
from src.Calculators.casters import THE_TRIPLE_CASTER
from src.Optical_Depth.opacity_table import opacity
import healpy as hp

# Snapshots of the simulation which we will use
fixes = np.arange(232,263 + 1)
fixes = [844, 881, 925, 950]
# For isotropic observers, set Healpy = True, otherwise false.
healpy = True
NSIDE = 4 # 192 observers
m = 6 # Select black hole, 4 or 6
#%% Constants & Converter
G = 6.6743e-11 # SI
Msol = 1.98847e30 # kg
Rsol = 6.957e8 # m
t = np.sqrt(Rsol**3 / (Msol*G )) # Follows from G=1
Mbh = 10**m # * Msol
Rt =  Mbh**(1/3) # Msol = 1, Rsol = 1

# Need these for the PW potential
c = 3e8 * t/Rsol # c in simulator units.
rg = 2*Mbh/c**2
c_cgs = 3e10

# Density Converter
Msol_to_g = 1.989e33
Rsol_to_cm = 6.957e10
den_converter = Msol_to_g / Rsol_to_cm**3

# Energy / Mass to cgs converter
energy_converter =  Msol_to_g * Rsol_to_cm**2 / (t**2)

# Energy Denstiy converter
en_den_converter = Msol_to_g / (Rsol_to_cm  * t**2 )

#%% Data Load
def doer_of_thing(fix, m):
    fix = str(fix)
    if m == 4:
        folder = '4/'
    else:
        folder = ''
        
    X = np.load( folder +fix + '/CMx_' + fix + '.npy')
    Y = np.load( folder +fix + '/CMy_' + fix + '.npy')
    Z = np.load( folder +fix + '/CMz_' + fix + '.npy')
    Mass = np.load( folder +fix + '/Mass_' + fix + '.npy')
    Den = np.load( folder +fix + '/Den_' + fix + '.npy')
    Rad = np.load( folder +fix + '/Rad_' + fix + '.npy')
    # if m == 6:
    #     Vol = np.load( folder +fix + '/Volume_' + fix + '.npy')
    # else:
    #     Vol = np.load( folder +fix + '/Vol_' + fix + '.npy')
        
    # # Average grid distance needs to be similar and slighltly bigger than 
    # # cell radius
    # cell_radii = (3 * Vol/4)**(1/3)
    # print(np.mean(cell_radii))
    T = np.load( folder +fix + '/T_' + fix + '.npy')

    # Convert Energy / Mass to Energy Density 
    Rad *= Den 
    Rad *= en_den_converter # to cgs
    Den *= den_converter # to cgs
    
    # Convert to spherical
    R, THETA, PHI = cartesian_to_spherical(X,Y,Z)
    R = R.value 
    THETA = THETA.value
    PHI = PHI.value
    
    # NOTE: Non-uniform observers, use healpix or sample in cosθ
    start = Rt
    stop = 500 * Rt
    if m ==6:
        num = 500 # about the average of cell radius
    if m == 4:
        num = 350
    radii = np.linspace(start, stop, num = 500)
    
    if healpy:
        thetas = np.zeros(192)
        phis = np.zeros(192)
        for i in range(0,192):
           thetas[i], phis[i] = hp.pix2ang(NSIDE, i)
           thetas[i] -= np.pi/2
           phis[i] -= np.pi
    else:     
        t_num = 7
        p_num = 16
        thetas = np.linspace(-np.pi/2, np.pi/2, num = t_num) 
        phis = np.linspace(0, 2 * np.pi, num = p_num)
        
    #%% Cast
    Rad_casted = THE_TRIPLE_CASTER(radii, R, thetas, THETA, phis, PHI,
                      Rad, 
                      weights = Mass, avg = True)
    Den_casted = THE_TRIPLE_CASTER(radii, R, thetas, THETA, phis, PHI,
                      Den,
                      weights = Mass, avg = True) 
    T_casted = THE_TRIPLE_CASTER(radii, R, thetas, THETA, phis, PHI,
                      T, 
                      weights = Mass, avg = True)
    Rad_casted = np.nan_to_num(Rad_casted)
    
    #%% Make Rays
    rays = []
    rays_den = []
    rays_T = []
    for i, theta in enumerate(thetas):
        for j, phi in enumerate(phis):
            # Ray holds Erad
            rays.append(Rad_casted[: , i , j])
            
            # The Density in each ray
            d_ray = Den_casted[:, i , j]
            d_ray = np.log10(d_ray)
            d_ray = np.nan_to_num(d_ray, neginf = 0)
            rays_den.append(d_ray)
            
            # The Temperature in each ray
            t_ray = T_casted[:, i , j]
            t_ray = np.log10(t_ray)
            t_ray = np.nan_to_num(t_ray, neginf = 0)
            rays_T.append(t_ray)
    
    #%% Let's see how it looks 
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
    cbar.set_label('Density')
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
                
            k_ross = opacity(Density, Temperature)
            
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
    
    if m==6:
        sphere_radius = 35_000
    else:
        sphere_radius = 7_000
    grad_E, radius_idx = grad_calculator(rays, radii, sphere_radius)
    flux = flux_calculator(grad_E, radius_idx, 
                           rays, rays_T, rays_den)
    
    # Divide by number of observers
    if healpy:
        flux = np.sum(flux) / 192
    else:
        flux = np.sum(flux) / (t_num * p_num)
    
    # Turn to luminosity
    sphere_radius *= Rsol_to_cm
    lum = flux * 4 * np.pi * sphere_radius**2
    print('%.2e' % lum )
    return lum

lums = []
for fix in fixes:
    lum = doer_of_thing(fix, m)
    lums.append(lum)
from src.Utilities.finished import finished
finished()
#%%
plt.figure()
if m == 4:
    days = [4.02,4.06,4.1,4.13,4.17,4.21,4.24,4.28,4.32,4.35,4.39,4.43,4.46,4.5,4.54,4.57,4.61,4.65,4.68,4.72,4.76,4.79,4.83,4.87,4.91,4.94,4.98,5.02,5.05,5.09,5.13,5.16]
else:
    days = [40, 45, 52, 55]
plt.plot(days, lums, '-o', color = 'maroon')
plt.yscale('log')
plt.ylim(1e41,1e45)
plt.ylabel('Bolometric Luminosity [erg/s]')
plt.xlabel('Days')
plt.title('FLD for $10^4 \quad M_\odot$')
plt.grid()
