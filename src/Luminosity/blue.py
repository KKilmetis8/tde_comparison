#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 16:02:14 2023

@author: konstantinos, paola 

Calculate the bolometric luminosity in the blued (BB) curve.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.Luminosity.emissivity import emissivity
from src.Luminosity.photosphere import get_photosphere

# Choose BH
m = 6
fixes = [844, 881, 925, 950]

# Constants
c = 2.9979e10 #[cm/s]
h = 6.6261e-27 #[gcm^2/s]
Kb = 1.3806e-16 #[gcm^2/s^2K]

Rsol_to_cm = 6.957e10

def planck_fun_n_cell(n,T):
    const = 2*h/c**2
    fun = const * n**3 / (np.exp(h*n/(Kb*T))-1)
    return fun

def planck_fun_cell(T):
    n_min = 4.8e14 #[Hz]
    n_max = 1e15 #[Hz]
    n_array = np.linspace(n_min,n_max, 1000)
    planck_fun_n_array = planck_fun_n_cell(n_array,T)
    fun = np.trapz(planck_fun_n_array, n_array)
    return fun

def luminosity(Temperature, Density, tau, volume):
    lum = 0
    for i in range(len(tau)):              
        T = Temperature[-i] # Out to in
        rho = Density[-i] 
        cell_vol = volume[-i]
        
        opt_depth = tau[i]
        # Ensure we can interpolate
        logT = np.log(T)
        logT = np.nan_to_num(logT, nan = 0, posinf = 0, neginf= 0)
        logrho = np.log(rho)
        logrho = np.nan_to_num(logrho, nan = 0, posinf = 0, neginf= 0)

        if logrho < -22 or  logT < 1 or logT > 17.8:
            continue
        
        epsilon = emissivity(T, rho, cell_vol)
        lum_cell = epsilon * planck_fun_cell(T) * np.exp(-opt_depth)
        lum += lum_cell
    return lum

if __name__ == "__main__":
    lums = []
    for fix in fixes:
        rays_den, rays_T, rays_tau, photosphere, radii = get_photosphere(fix, m)
        dr = (radii[1] - radii[0]) * Rsol_to_cm
        volume = 4 * np.pi * radii**2 * dr  / 480 # fix when we get 192
        
        global_lum = 0 
        for i, ray in  enumerate(rays_den):
            global_lum += luminosity(rays_T[i], rays_den[i], rays_tau[i], volume)
        lums.append( np.log10(global_lum))
    
    days = [40, 45, 52, 55]
    plt.plot(days, lums, 'o-', c = 'cornflowerblue')
