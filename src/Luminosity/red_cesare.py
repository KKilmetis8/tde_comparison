#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 16:42:47 2023

@author: paola

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
import numba
# Custom Imports
from src.Opacity.opacity_table import opacity
from src.Calculators.ray_cesare import ray_maker
from src.Luminosity.special_radii import calc_photosphere
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
        snapshots = [844, 881, 898, 925, 950] #[844, 881, 882, 898, 925, 950]
        days = [1, 1.1, 1.3, 1.4] #[1, 1.139, 1.143, 1.2, 1.3, 1.4] # t/t_fb
        const = 0.05
        beginning = 1200
    num_array = beginning * np.ones(len(snapshots))
    for i in range(1,len(num_array)):
            num_array[i] = int(1.5 * num_array[i-1])
    return snapshots, days, num_array

@numba.njit
def grad_calculator(ray: np.array, radii: np.array, sphere_radius: int): 
    # For a single ray (in logspace) get 
    # the index of radius closest to sphere_radius and the gradE there.
    # Everything is in CGS.
    for i, radius in enumerate(radii):
        if radius > sphere_radius:
            idx = i - 1 
            break
        
    step = radii[idx+1] - radii[idx]
    
    grad_E = (ray[idx+1] - ray[idx]) / step 

    return grad_E, idx

    
def flux_calculator(grad_E, idx_tot, 
                    rays, rays_T, rays_den):
    """
    Get the flux for every observer.
    Eevrything is in CGS.

    Parameters: 
    grad_E idx_tot are 1D-array of lenght = len(rays)
    rays, rays_T, rays_den are len(rays) x N_cells arrays
    """
    f = np.zeros(len(grad_E))
    max_count = 0
    max_but_zero_count = 0
    zero_count = 0
    flux_zero = 0
    flux_count = 0
    for i, ray in enumerate(rays):
        # We compute stuff OUTSIDE the photosphere
        # (which is at index idx_tot[i])
        idx = idx_tot[i]+1 #  
        Energy = ray[idx]
        max_travel = np.sign(-grad_E[i]) * c_cgs * Energy # or should we keep the abs???
        
        Temperature = rays_T[i][idx]
        Density = rays_den[i][idx]
        
        # Ensure we can interpolate
        rho_low = np.exp(-45)
        T_low = np.exp(8.77)
        T_high = np.exp(17.8)
        
        # If here is nothing, light continues
        if Density < rho_low:
            max_count += 1
            f[i] = max_travel
            if max_travel == 0:
                max_but_zero_count +=1
            continue
        
        # If stream, no light 
        if Temperature < T_low: 
            zero_count += 1
            f[i] = 0 
            continue
        
        # T too high => Kramers'law
        if Temperature > T_high:
            X = 0.7389
            k_ross = 3.68 * 1e22 * (1 + X) * Temperature**(-3.5) * Density #Kramers' opacity [cm^2/g]
            k_ross *= Density
        else:    
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
            if max_travel == 0:
                max_but_zero_count +=1
        else:
            flux_count += 1
            f[i] = Flux
            if Flux == 0:  
                flux_zero += 1

    print('Max: ', max_count)
    print('Zero due to: \n - max travel: ', max_but_zero_count)
    print('- T_low:', zero_count)
    print('- flux:', flux_zero) 
    print('Flux: ', flux_count)
    return f

def doer_of_thing(fix, m, num):
    rays_T, rays_den, rays, radii = ray_maker(fix, m, num)

    grad_E_tot = []
    idx_tot = []
    sphere_radius = []
    for i in range(len(rays_T)):
    # Calculate gradE for every ray
        temp = rays_T[i]
        dens = rays_den[i]
        ray = rays[i]
        _, _, photo = calc_photosphere(temp, dens, radii)
        sphere_radius.append(photo)
        grad_E, idx = grad_calculator(ray, radii, photo)
        grad_E_tot.append(grad_E)
        idx_tot.append(idx)

    # Calculate Flux and see how it looks
    flux = flux_calculator(grad_E_tot, idx_tot, 
                            rays, rays_T, rays_den)
    plt.figure(figsize = [8,5])
    plt.scatter(np.arange(192), flux, s = 3, color = 'orange')     
    plt.xlabel('Observer')
    plt.ylabel(r'Flux [erg/s cm$^2$]')
    plt.grid()
    plt.savefig('Figs/' + str(fix) + '_NEWflux.png')                   

    # Calculate luminosity 
    lum = np.zeros(len(flux))
    zero_count = 0
    for i in range(len(flux)):
        # Turn to luminosity
        if flux[i] == 0:
            zero_count += 1
        if flux[i] < 0:
            flux[i] = 0 
        lum[i] = flux[i] * 4 * np.pi * sphere_radius[i]**2

    # Average in observers
    lum = np.sum(lum)/192
    print('tot zeros:', zero_count)
    print('Fix %i' %fix, ', Lum %.3e' %lum )
    return lum
#%%
##
# MAIN
##
if __name__ == "__main__":
    save = True
    plot = True
    m = 6 # Choose BH
    fixes, days, num_array = select_fix(m)
    lums = []
            
    for idx,fix in enumerate(fixes):
        lum = doer_of_thing(fix, m, int(num_array[idx]))
        lums.append(lum)
    
    if save:
        np.savetxt('data/new2_reddata_m'+ str(m) + '.txt', (days, lums)) 
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
        plt.savefig('Final plot/new2_red' + str(m) + '.png')
        plt.show()

