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

n_min = 1e12 # [Hz] minimum frequency for integration
n_max = 1e18 # [Hz] maximum frequency for integration
n_spacing = 1000
n_array = np.linspace(n_min,n_max, n_spacing)

#Wien law
const_npeak = 5.879e10 #[Hz/K]

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

def find_peak(Temperature):
    """Find n peak with Wien law."""
    npeak = const_npeak * Temperature
    return 10*npeak

def planck_fun_n_cell(Temperature: float, n: float) -> float:
    """ Planck function in a cell. """
    const = 2*h/c**2
    peak = find_peak(Temperature)
    if n> 10*peak:
        fun = 0
    else:
        fun = const * n**3 / (np.exp(h*n/(Kb*Temperature))-1)
    return fun

def planck_fun_cell(Temperature: float) -> float:
    """
    Bolometric planck function in a cell. 
    We select the range for frequency in order to no overcome the peak or we have a mess.
    """
    planck_fun_n_array = []
    peak = find_peak(Temperature)
    n_arr = np.linspace(n_min,peak,n_spacing)
    for n in n_arr:
        planck_fun_n_array_single = planck_fun_n_cell(Temperature, n)
        planck_fun_n_array.append(planck_fun_n_array_single)
    planck_fun_n_array = np.array(planck_fun_n_array)
    fun = np.trapz(planck_fun_n_array, n_arr)
    return fun

def luminosity_n(Temperature: float, Density: float, tau: float, volume: float, n:int):
    """
    Temperature, Density and volume: np.array from near to the BH to far away. Thus we will use negative index in the for loop.
    tau: np.array from outside to inside.
    n is the frequency.

    We obtain luminosity in a ray as function of frequency.
    """
    # lum = 0
    # for i in range(len(tau)):              
    #     T = Temperature[-i]
    #     rho = Density[-i] 
    #     opt_depth = tau[i]
    #     cell_vol = volume[-i]

    #     # Ensure we can interpolate
    #     rho_low = np.exp(-22)
    #     T_low = np.exp(8.77)
    #     T_high = np.exp(17.8)
    #     if rho < rho_low or T < T_low or T > T_high:
    #         continue
        
    epsilon = emissivity(Temperature, Density, volume)
    lum_cell = epsilon * planck_fun_n_cell(Temperature, n) * np.exp(-tau)
    return (lum_cell/planck_fun_cell(Temperature))

def luminosity(Temperature: float, Density: float, tau: float, volume: float) -> int:
    """Gives NOT normalised bolometric luminosity in a ray."""
    lum_n_array = []
    peak = find_peak(Temperature)
    n_arr = np.linspace(n_min,peak,n_spacing)
    for n in n_arr:
        value = luminosity_n(Temperature, Density, tau, volume, n)
        lum_n_array.append(value)
    lum_n_array = np.array(lum_n_array)
    lum = np.trapz(lum_n_array, n_arr)
    return lum 
    
# def normalised_luminosity_n(Temperature: float, Density: float, tau: float, volume: float, n: float, luminosity_fld: float):
#     """
#     luminosity_fld: float. It's the luminosity with FLD method from the considered snapshot.
#     Gives the luminosity normalised with FLD model. """
#     norm = luminosity_fld / luminosity(Temperature, Density,  tau, volume)
#     value = luminosity_n(Temperature, Density,  tau, volume, n)
#     # value = np.nan_to_num(value) #maybe we don't need this beacuse we've selected the range of freuency
#     return value * norm

if __name__ == "__main__":
    
    #CHECK PLANCK
    # check_planck_n = []
    # for n in n_array:
    #     a = planck_fun_n_cell(1e6, n)
    #     check_planck_n.append(np.log10(a))
    # plt.plot(np.log10(n_array),np.array(check_planck_n))
    # plt.xlim(8,18)

    fix = 233
    fld_data = np.loadtxt('reddata_m'+ str(m) +'.txt')
    luminosity_fld_fix = fld_data[1]
    rays_den, rays_T, rays_tau, photosphere, radii = get_photosphere(fix, m)
    dr = (radii[1] - radii[0]) * Rsol_to_cm
    volume = 4 * np.pi * radii**2 * dr  / 192
    print(np.array(rays_T).shape) 
    print(np.array(volume).shape)
    # for n in n_array:
    #     a = normalised_luminosity_n(1e4, 1e-10,1,1e-5,n,10)
    #     print(a)

    lum_tilde_n = np.zeros(len(n_array))
    for j in range(len(rays_den)):
        for i in range(len(rays_tau[j])):              
            T = rays_T[j][-i]
            rho = rays_den[j][-i] 
            opt_depth = rays_tau[j][i]
            cell_vol = volume[-i]

            # Ensure we can interpolate
            rho_low = np.exp(-22)
            T_low = np.exp(8.77)
            T_high = np.exp(17.8)
            if rho < rho_low or T < T_low or T > T_high:
                continue

            norm = luminosity_fld_fix[0] * luminosity(T, rho, opt_depth, cell_vol)
            for i in range(len(n_array)):
                lum_nu_cell = luminosity_n(T, rho, opt_depth, cell_vol, n_array[i])
                lum_tilde_n[i] += lum_nu_cell
        print(j)

    # lum_tilde_n = []
    # for n in n_array:
    #     lum_rays = 0 
    #     for j in range(len(rays_den)):
    #         for i in range(len(rays_tau[j])):              
    #             T = rays_T[j][-i]
    #             rho = rays_den[j][-i] 
    #             opt_depth = rays_tau[j][i]
    #             cell_vol = volume[-i]

    #             # Ensure we can interpolate
    #             rho_low = np.exp(-22)
    #             T_low = np.exp(8.77)
    #             T_high = np.exp(17.8)
    #             if rho < rho_low or T < T_low or T > T_high:
    #                 continue

    #             #luminosity in a cell
    #             lum = normalised_luminosity_n(T,  rho, opt_depth, 
    #                                     cell_vol, n, luminosity_fld_fix[0])
    #             lum_rays += lum
    #         print(j)
    #     lum_tilde_n.append(lum_rays)
    plt.plot(np.log10(n_array), np.log10(lum_tilde_n))
    plt.show()

