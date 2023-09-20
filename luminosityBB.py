#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 16:02:14 2023

@author: konstantinos, paola 

Calculate the luminosity NOT normalized that we will use in the blue (BB) curve.

NOTES FOR OTHERS:
- arguments are in CGS
- temperature and density have to be in log10(CGS) scale ONLY in the functions: luminosity_n, luminosity and normalised_luminosity_n
- make changes in VARIABLES: m (power index of the BB mass), fixes (number of snapshots) anf thus days
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

n_min = 1e2 # [Hz] minimum frequency for integration
n_max = 1e24 # [Hz] maximum frequency for integration
n_array = np.linspace(n_min,n_max, 1000)

###
##
# VARIABLES
##
###

m = 4 # Choose BH
if m == 4:
    fixes = np.arange(233,263 + 1)
    days = [1.015, 1.025, 1.0325, 1.0435, 1.0525, 1.06, 1.07, 1.08, 1.0875, 1.0975, 1.1075, 1.115, 1.125, 1.135, 1.1425, 1.1525, 1.1625, 1.17, 1.18, 1.19, 1.1975, 1.2075, 1.2175, 1.2275, 1.235, 1.245, 1.255, 1.2625, 1.2725, 1.2825, 1.29] #t/t_fb
    # days = [4.06,4.1,4.13,4.17,4.21,4.24,4.28,4.32,4.35,4.39,4.43,4.46,4.5,4.54,4.57,4.61,4.65,4.68,4.72,4.76,4.79,4.83,4.87,4.91,4.94,4.98,5.02,5.05,5.09,5.13,5.16] #days
if m == 6:
    fixes = [844, 881, 925, 950]
    days = [1.00325, 1.13975, 1.302, 1.39425] #t/t_fb
    # days = [40, 45, 52, 55] #days

###
##
# FUNCTIONS
##
###

def emissivity(Temperature, Density, cell_vol):
    """Arguments in CGS. Gives emissivity in a cell. """
    k_planck = opacity(Temperature, Density, 'planck', ln = False)
    emiss = alpha * c * Temperature**4 * k_planck * cell_vol
    return emiss

def planck_fun_n_cell(Temperature, n):
    """ Planck function in a cell. """
    const = 2*h/c**2
    fun = const * n**3 / (np.exp(h*n/(Kb*Temperature))-1)
    return fun

def planck_fun_cell(Temperature):
    """Bolometric planck function in a cell."""
    planck_fun_n_array = planck_fun_n_cell(Temperature, n_array)
    fun = np.trapz(planck_fun_n_array, n_array)
    return fun

def luminosity_n(Temperature, Density, tau, volume, n):
    """
    Temperature, Density and volume: np.array from near to the BH to far away. Thus we will use negative index in the for loop.
    tau: np.array from outside to inside.
    All these array have the same shape and are in CGS.
    n is the frequency.

    We obtain luminosity as function of frequency.
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
        lum_cell = epsilon * planck_fun_n_cell(T, n) * np.exp(-opt_depth)
        lum += lum_cell
    return (lum/planck_fun_cell(T))

def luminosity(Temperature, Density,  tau, volume):
    """Gives NOT normalised bolometric luminosity."""
    lum_n_array = luminosity_n(Temperature, Density, tau, volume, n_array)
    print(lum_n_array.shape)
    print(n_array.shape)
    lum = np.trapz(lum_n_array, n_array)
    return lum 
    
def normalised_luminosity_n(Temperature, Density,  tau, volume, n, luminosity_fld):
    """
    luminosity_fld: float. It's the luminosity with FLD method from the considered snapshot.
    Gives the luminosity normalised with FLD model. """
    value = luminosity_n(Temperature, Density, tau, volume, n) * luminosity_fld / luminosity(Temperature, Density,  tau, volume)
    return value

if __name__ == "__main__":
    fix = 233
    fld_data = np.loadtxt('reddata_m'+ str(m) +'.txt')
    luminosity_fld_fix = fld_data[1]
    rays_den, rays_T, rays_tau, photosphere, radii = get_photosphere(fix, m)
    dr = (radii[1] - radii[0]) * Rsol_to_cm
    volume = 4 * np.pi * radii**2 * dr  / 192 # fix when we get 192
    
    global_lum = 0 
    for i, ray in  enumerate(rays_den):
        lum = luminosity(rays_T[i], rays_den[i],  rays_tau[i], volume)        
        #lum = normalised_luminosity_n(rays_T[i], rays_den[i],  rays_tau[i], 
                                    # volume, n_array, luminosity_fld_fix[0])
        global_lum += lum
    print(global_lum)

    plt.plot(global_lum, n_array)
    plt.show()
#%% Plotting
    # plt.rcParams['text.usetex'] = True
    # plt.rcParams['figure.dpi'] = 300
    # plt.rcParams['figure.figsize'] = [5 , 3]
    # plt.rcParams['axes.facecolor'] = 'whitesmoke'
    
    # plt.plot(days,np.log10(lums), 'o-', c = 'royalblue')
    # np.savetxt('bluedata_m' + str(m) + '.txt', (days, lums))
    # plt.title(r'$10^' + str(m) + ' M_\odot$ BB Fit')
    # plt.xlabel('Days')
    # plt.ylabel('Bolometric Luminosity $log_{10}(L)$ $[L_\odot]$')
    # plt.grid()
    # plt.show()
    # # plt.savefig('plot.png')

