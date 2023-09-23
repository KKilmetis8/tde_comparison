#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 16:02:14 2023

@author: konstantinos, paola 

Calculate the luminosity NOT normalized that we will use in the blue (BB) curve.

NOTES FOR OTHERS:
- All the functions have to be applied to a CELL
- arguments are in cgs, NOT in log.
- make changes in VARIABLES: frequencies range, fixes (number of snapshots) anf thus days
"""

# Vanilla imports
import numpy as np
import matplotlib.pyplot as plt
import numba

# Chocolate imports
from src.Luminosity.photosphere import get_photosphere
from src.Optical_Depth.opacity_table import opacity

# Constants
c = 2.9979e10 #[cm/s]
h = 6.6261e-27 #[gcm^2/s]
Kb = 1.3806e-16 #[gcm^2/s^2K]
alpha = 7.5646 * 10**(-15) # radiation density [erg/cm^3K^4]
Rsol_to_cm = 6.957e10

# Wien law
const_npeak = 5.879e10 #[Hz/K]

###
##
# VARIABLES
##
###

# Frequencies [Hz]
n_min = 1e12 
n_max = 1e18 
n_spacing = 1000
n_array = np.logspace(np.log10(n_min),np.log10(n_max), num = n_spacing)

# Snapshots
fixes4 = np.arange(233,263 + 1)
days4 = [1.015, 1.025, 1.0325, 1.0435, 1.0525, 1.06, 1.07, 1.08, 1.0875, 1.0975, 1.1075, 1.115, 1.125, 1.135, 1.1425, 1.1525, 1.1625, 1.17, 1.18, 1.19, 1.1975, 1.2075, 1.2175, 1.2275, 1.235, 1.245, 1.255, 1.2625, 1.2725, 1.2825, 1.29] #t/t_fb
# days4 = [4.06,4.1,4.13,4.17,4.21,4.24,4.28,4.32,4.35,4.39,4.43,4.46,4.5,4.54,4.57,4.61,4.65,4.68,4.72,4.76,4.79,4.83,4.87,4.91,4.94,4.98,5.02,5.05,5.09,5.13,5.16] #days
fixes6 = [844, 881, 925, 950]
days6 = [1.00325, 1.13975, 1.302, 1.39425] #t/t_fb
# days6 = [40, 45, 52, 55] #days

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
    
def normalisation(Temperature: float, Density: float, tau: float, volume: float, luminosity_fld: float) -> float:
    """ Find the normalisation constant from FLD model. """      
    norm = luminosity_fld / luminosity(Temperature, Density, tau, volume)
    print('lum',luminosity(Temperature, Density, tau, volume))
    print('norm',norm)
    return  norm

######
# MAIN
#####
if __name__ == "__main__":
    m = 6
    #CHECK PLANCK
    # check_planck_n = []
    # for n in n_array:
    #     a = planck_fun_n_cell(1e6, n)
    #     check_planck_n.append(a)
    # plt.plot(n_array,check_planck_n)
    # plt.loglog()
    
    fix = 844
    fld_data = np.loadtxt('reddata_m'+ str(m) +'.txt')
    luminosity_fld_fix = fld_data[1]
    rays_den, rays_T, rays_tau, photosphere, radii = get_photosphere(fix, m)
    dr = (radii[1] - radii[0]) * Rsol_to_cm
    volume = 4 * np.pi * radii**2 * dr  / 192
    print(np.array(rays_T).shape) 
    print(np.array(volume).shape)

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

            norm = normalisation(T, rho, opt_depth, cell_vol, luminosity_fld_fix[0])
            for i in range(len(n_array)):
                lum_nu_cell = luminosity_n(T, rho, opt_depth, cell_vol, n_array[i]) * norm
                lum_tilde_n[i] += lum_nu_cell
            print(lum_nu_cell)
            print('rho:',rho)
        print(j)
    
    test = 0
    for n in range(len(lum_tilde_n)):
        test += lum_tilde_n[i]
    print('bolometric luminosity:', test)

    plt.figure()
    plt.plot(n_array, lum_tilde_n)
    plt.loglog()
    
    plt.xlabel(r'log$\nu$ [Hz]')
    plt.ylabel(r'log$\tilde{L}_\nu$ [erg/s]')
    #plt.title(f'$10^{str(m)}$ BH snap ' + fix )
    plt.grid()
    plt.savefig('Ltilda_m' + str(m) + '_snap' + str(fix))
    plt.show()

