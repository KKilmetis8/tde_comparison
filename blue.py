#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 16:02:14 2023

@author: konstantinos, paola 

Calculate the bolometric luminosity in the blue (BB) curve.

NOTES FOR OTHERS:
- make changes in variables: m (power index of the BB mass), fixes (number of snapshots) anf thus days
"""

# Vanilla imports
import numpy as np
import matplotlib.pyplot as plt

# Chocolate imports
from src.Luminosity.photosphere import get_photosphere
from src.Optical_Depth.opacity_table import opacity

# Constants
c = 2.9979e10 #[cm/s]
h = 6.6261e-27 #[gcm^2/s]
Kb = 1.3806e-16 #[gcm^2/s^2K]
alpha = 7.5646 * 10**(-15) # radiation density [erg/cm^3K^4]
Rsol_to_cm = 6.957e10

###
##
# VARIABLES
##
###

m = 6 # Choose BH
if m == 4:
    fixes = np.arange(233,263 + 1)
    days = [1.015, 1.025, 1.0325, 1.0435, 1.0525, 1.06, 1.07, 1.08, 1.0875, 1.0975, 1-1075, 1.115, 1.125, 1.135, 1.1425, 1.1525, 1.1625, 1.17, 1.18, 1.19, 1.1975, 1.2075, 1.2175, 1.2275, 1.235, 1.245, 1.255, 1.2625, 1.2725, 1.2825, 1.29] #t/t_fb
    #days = [4.06,4.1,4.13,4.17,4.21,4.24,4.28,4.32,4.35,4.39,4.43,4.46,4.5,4.54,4.57,4.61,4.65,4.68,4.72,4.76,4.79,4.83,4.87,4.91,4.94,4.98,5.02,5.05,5.09,5.13,5.16] #days
if m == 6:
    fixes = [844, 881, 925, 950]
    days = [1.00325, 1.13975, 1.302, 1.39425] #t/t_fb
    #days = [40, 45, 52, 55] #days

###
##
# FUNCTIONS
##
###

def emissivity(T, rho, cell_vol):
    """ Arguments in CGS. NB: T and rho DON'T have to be in log scale"""
    k_planck = opacity(T, rho, 'planck', ln = False)
    emiss = alpha * c * T**4 * k_planck * cell_vol
    return emiss

def planck_fun_n_cell(n,T):
    const = 2*h/c**2
    fun = const * n**3 / (np.exp(h*n/(Kb*T))-1)
    return fun

def planck_fun_cell(T):
    n_min = 4.8e14 # [Hz]
    n_max = 1e15 # [Hz]
    n_array = np.linspace(n_min,n_max, 1000)
    planck_fun_n_array = planck_fun_n_cell(n_array,T)
    fun = np.trapz(planck_fun_n_array, n_array)
    return fun

def luminosity(Temperature, Density,  tau, volume):
    """
    Temperature, Density and volume: np.array from near to the BH to far away. Thus we will use negative index in the for loop.
    tau: np.array from outside to inside.
    All these array have the same shape.
    """
    lum = 0
    for i in range(len(tau)):              
        T = Temperature[-i]
        rho = Density[-i] 
        opt_depth = tau[i]
        cell_vol = volume[-i]

        # Ensure we can interpolate
        rho_low = np.exp(-22)
        T_low = np.exp(8.77)
        T_high = np.exp(17.8)
        if rho < rho_low or T < T_low or T > T_high:
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
        volume = 4 * np.pi * radii**2 * dr  / 192 # fix when we get 192
        
        global_lum = 0 
        for i, ray in  enumerate(rays_den):
            lum = luminosity(rays_T[i], rays_den[i],  rays_tau[i], 
                                     volume)
            global_lum += lum
        lums.append( np.log10(global_lum))
    print(lums)
#%% Plotting
    plt.rcParams['text.usetex'] = True
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['figure.figsize'] = [5 , 3]
    plt.rcParams['axes.facecolor'] = 'whitesmoke'
    
    
    #plt.ylim(41.5,45.5)
    #plt.xlim(39,56)
    plt.plot(days, lums, 'o-', c = 'royalblue')
    plt.title(r'$10^' + str(m) + ' M_\odot$ BB Fit')
    plt.xlabel('Days')
    plt.ylabel('Bolometric Luminosity $log_{10}(L)$ $[L_\odot]$')
    plt.grid()
    plt.show()
    plt.savefig('plot.png')

