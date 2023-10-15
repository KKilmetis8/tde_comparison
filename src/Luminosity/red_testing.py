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
# Vanilla Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numba
import colorcet
# Custom Imports
from src.Calculators.ray_maker import ray_maker
from src.Opacity.opacity_table import opacity
# from src.Luminosity.photosphere import get_photosphere
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [5 , 4]

# Constants & Converter
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
        snapshots = [844] # 881,] # 925, 950]
        days = [1] # 1.1,]# 1.3, 1.4] #t/t_fb
    return snapshots, days

### OLD
# if m == 4:
#     fixes =  np.arange(233,263+1) #from 233 to 263
#     days = [1.015, 1.025, 1.0325, 1.0435, 1.0525, 1.06, 1.07, 1.08, 1.0875, 1.0975, 1.1075, 1.115, 1.125, 1.135, 1.1425, 1.1525, 1.1625, 1.17, 1.18, 1.19, 1.1975, 1.2075, 1.2175, 1.2275, 1.235, 1.245, 1.255, 1.2625, 1.2725, 1.2825, 1.29] #t/t_fb
# if m == 6:
#     fixes = [844] #[844, 881, 925, 950] # 1006]
#     days = [1.00325, 1.13975, 1.302, 1.39425] # 1.6] #t/t_fb
###


@numba.njit
def grad_calculator(rays, radii, sphere_radius = 15_000): 
    # Get the index of radius closest in sphere radius
    # diffs = np.abs(radii - sphere_radius)
    # idx = np.argmin(diffs)
    for i, radius in enumerate(radii):
        if radius > sphere_radius:
            idx = i - 1 
            break
        
    step = radii[1] - radii[0]
    grad_E = np.zeros(len(rays))
    
    for i, ray in enumerate(rays):
        grad_E[i] = (ray[idx+1] - ray[idx]) / step 
        
    return grad_E, idx

# @numba.njit
def converger(rays, radii):
    
    grad_Es = []
    idxs = []
    for sphere_radius in radii:
        grad_E, idx = grad_calculator(rays, radii, sphere_radius)
        grad_Es.append(grad_E)
        idxs.append(idx)
        
    rel_error = [ 100 * (1 - (grad_Es[i]/grad_Es[i-1])) 
                 for i in range(1, len(grad_Es))]
    
    plt.figure( figsize = (16,4))
    plt.plot(radii[1::10], rel_error[::10], '-o', c='k')
    plt.xlabel('Sphere Radii')
    plt.ylabel('Relative Error')
    plt.ylim(-25, 25)
    return grad_Es, idxs
    
def flux_calculator(grad_E, idx, 
                    rays, rays_T, rays_den):
    f = np.zeros(len(grad_E))
    max_count = 0
    zero_count = 0
    flux_count = 0
    idx += 1 # Outside the sphere
    for i, ray in enumerate(rays):
        # Get opacity
        Energy = ray[idx]
        max_travel = c_cgs * Energy
        
        Temperature = rays_T[i][idx]
        Density = rays_den[i][idx]
        
        # Ensure we can interpolate
        rho_low = np.exp(-22)
        T_low = np.exp(8.77)
        T_high = np.exp(17.8)
        
        # If here is nothing, light continues
        if Density < rho_low:
            zero_count += 1
            f[i] = max_travel
            continue
        
        if Temperature < T_low:
            continue
        
        # T too high => Thompson opacity
        if Temperature > T_high:
            print('hiiiiiiiiiii')
            Temperature = np.exp(17.7)
            
        # Get Opacity, NOTE: Breaks Numba
        k_ross = opacity(Temperature, Density, 'rosseland', ln = False)
        
        # Calc R, eq. 28
        R = np.abs(grad_E[i]) /  (k_ross * Energy)
        invR = 1 / R
        
        # Calc lambda, eq. 27
        coth = 1 / np.tanh(R)
        lamda = invR * (coth - invR)
        # Calc Flux, eq. 26
        Flux = - c_cgs * grad_E[i]  * lamda / k_ross
        
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

#%%
##
# MAIN
##
if __name__ == "__main__":
    plot = True
    
    lums = []
    m = 6 # Choose BH
    index = 0
    fixes, days = select_fix(m)
    fix = 844
    
    rays_T, rays_den, rays, radii = ray_maker(fix, m)
    #%%
    # Let's see how it looks
    if plot: 
        img = plt.pcolormesh(radii, np.arange(len(rays)), rays, cmap = 'cet_gouldian',
                             norm = colors.LogNorm())
        cbar = plt.colorbar(img)
        plt.title('Rays')
        cbar.set_label('Radiation Energy Density')
        plt.xlabel('r')
        plt.ylabel('Various observers')
        img.axes.get_yaxis().set_ticks([])
        plt.xlim(0,3000)
        # plt.xscale('log')
        
        plt.figure()
        img = plt.pcolormesh(radii, np.arange(len(rays)), rays_den, 
                             cmap = 'cet_fire', norm = colors.LogNorm())
        cbar = plt.colorbar(img)
        plt.title('Rays')
        cbar.set_label(r'$log_{10}$Density')
        plt.xlabel('r')
        plt.ylabel('Various observers')
        img.axes.get_yaxis().set_ticks([])
        plt.xlim(0,3000)
        # plt.xscale('log')
        
        plt.figure()
        img = plt.pcolormesh(radii, np.arange(len(rays)), rays_T, 
                             cmap = 'cet_bmy', norm = colors.LogNorm())
        cbar = plt.colorbar(img)
        plt.title('Rays')
        cbar.set_label('Temperature')
        plt.xlabel('r')
        plt.ylabel('Various observers')
        img.axes.get_yaxis().set_ticks([])
        plt.xlim(0,3000)
        # plt.xscale('log')
        
        # Calculate
        grad_Es, idxs = converger(rays, radii)
        # Calculate Flux
        # grad_E, radius_idx = grad_calculator(rays, radii, sphere_radius)
        lums = []
        for grad_E, idx, radius in zip(grad_Es, idxs, radii):
            flux = flux_calculator(grad_E, idx, 
                                    rays, rays_T, rays_den)
        
            # Divide by number of observers
            flux = np.sum(flux) / 192
        
            print('Flux %.3e' % flux )
            # Turn to luminosity
            lum = flux * 4 * np.pi * radius**2
            print('Lum %.3e' % lum )
            lums.append(lum)
        
        # Lums Plot
        plt.figure( figsize = (8,4))
        plt.plot(radii[1:-2] / Rsol_to_cm, lums[1:-2], c='maroon')
        plt.yscale('log')
        plt.ylabel('Luminosity [erg/s]')
        plt.xlabel('Sphere Radii [$R_\odot$]')
        plt.title('Flux Limited Diffusion')
        
        # Conv. Check
        rel_error = [ int(100 * (1 - (lums[i]/lums[i-1]))) for i in range(1, len(lums) -1)]
        plt.figure(figsize = (16,4))
        plt.title('Convergance Test')
        plt.plot(radii[1:-1], rel_error, '-o', c='k')
        plt.xlabel('Sphere Radii [$R_\odot$]')
        plt.ylabel('Relative Error in L')
        plt.ylim(-35, 35)
        # stop


