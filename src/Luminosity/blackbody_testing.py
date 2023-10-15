#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 16:42:10 2023

@author: konstantinos
"""

"""
Created on Mon Oct 9 2023

@author: konstantinos, paola 

Calculate the luminosity that we will use in the blue (BB) curve as sum of multiple BB.

NOTES FOR OTHERS:
- All the functions have to be applied to a CELL
- arguments are in cgs, NOT in log.
- make changes in VARIABLES: frequencies range, fixes (number of snapshots) anf thus days
"""

# Vanilla imports
import numpy as np
import matplotlib.pyplot as plt
AEK = '#F1C410'

# Chocolate imports
from src.Luminosity.photosphere import get_photosphere
from src.Opacity.opacity_table import opacity

# Constants
c = 2.99792458e10 #[cm/s]
h = 6.62607015e-27 #[gcm^2/s]
Kb = 1.380649e-16 #[gcm^2/s^2K]
alpha = 7.5646 * 10**(-15) # radiation density [erg/cm^3K^4]
Rsol_to_cm = 6.957e10
#%%
###
# FUNCTIONS
###

def log_array(n_min, n_max, lenght):
    x_min = np.log10(n_min)
    x_max = np.log10(n_max)
    x_arr = np.linspace(x_min, x_max , num = lenght)
    return x_arr

def select_fix(m):
    if m == 4:
        snapshots = [233, 254, 263, 277 , 293, 308, 322]
        days = [1, 1.2, 1.3, 1.4, 1.56, 1.7, 1.8] 
    if m == 6:
        snapshots = [881] #[844, 881, 925, 950]
        days = [1.1] # [1, 1.1, 1.3, 1.4] #t/t_fb
    return snapshots, days

def planck(Temperature: float, n: float) -> float:
    """ Planck function in a cell. It needs temperature and frequency. """
    
    const = 2*h/c**2
    fun = const * n**3 / (np.exp(h*n/(Kb*Temperature))-1)

    return fun

def luminosity_n(Temperature: float, Density: float, tau: float, volume: float, n:int):
    """ Luminosity in a cell: L_ni = \epsilon e^(-\tau) B_ni / B where  
    B = \sigma T^4/\pi"""
    k_planck = opacity(Temperature, Density, 'planck', ln = False)

    L = 4  * np.pi * k_planck * volume * np.exp(-tau) * planck(Temperature, n)
    # print('Bn: ', planck(Temperature, n))
    print('tau: ', np.exp(-tau))
    print('L:', L)
    return L

def normalisation(L_x: np.array, x_array: np.array, luminosity_fld: float) -> float:
    """ Given the array of luminosity L_x computed over 10^{x_array} (!!!), 
    find the normalisation constant from FLD model used for L_tilde_nu. """  
    xLx =  10**(x_array) * L_x
    L = np.trapz(xLx, x_array) 
    L *= np.log(10)
    norm = luminosity_fld / L
    return norm

#%% MAIN
if __name__ == "__main__":
    plot = False
    save = True
    
    # Choose BH and freq range
    m = 6
    n_min = 1e12 
    n_max = 1e20
    n_spacing = 10000
    x_arr = log_array(n_min, n_max, n_spacing)
    fix = 844
    
    # Save frequency range
    if save:
        with open('data/L_spectrum_m'+ str(m) + '.txt', 'w') as f:
            f.write('# exponents x of frequencues: n = 10^x  \n')
            f.write(' '.join(map(str, x_arr)) + '\n') 
    
    # Load data for normalization
    fld_data = np.loadtxt('data/reddata_m'+ str(m) +'.txt')
    luminosity_fld_fix = fld_data[1]
    n_arr = 10**x_arr
    
    # Get Photosphere
    rays_T, rays_den, rays_tau, photosphere, radii = get_photosphere(fix, m)
    dr = radii[1] - radii[0]
    volume = 4 * np.pi * radii**2 * dr  / 192
    #%%            
    lum_n = np.zeros(len(x_arr))

    for j in range(192):
        for i in range(len(rays_tau[j])):        
            # Temperature, Density and volume: np.array from near to the BH
            # to far away. 
            # Thus we will use negative index in the for loop.
            # tau: np.array from outside to inside.
            reverse_idx = -i -1
            T = rays_T[j][reverse_idx]
            rho = rays_den[j][reverse_idx] 
            opt_depth = rays_tau[j][i]
            cell_vol = volume[reverse_idx]
            # print('T:', T)
            # print('rho: ', rho)
            # Ensure we can interpolate
            rho_low = np.exp(-22)
            T_low = np.exp(8.77)
            T_high = np.exp(17.878)
            if T < T_low:
                continue
            if T > T_high:
                print('high')
                T = np.exp(17.87)         
            
            for i, n in enumerate(n_arr): #we need linearspace
                lum_n_cell = luminosity_n(T, rho, opt_depth, cell_vol, n)
                lum_n[i] += lum_n_cell
                
    # Normalise with the bolometric luminosity from red curve (FLD)
    const_norm = normalisation(lum_n, x_arr, luminosity_fld_fix[1])
    lum_tilde_n = lum_n * const_norm
            
    if plot:
        fig, ax = plt.subplots()
        ax.plot(n_arr, lum_tilde_n)
        plt.xlabel(r'$log_{10}\nu$ [Hz]')
        plt.ylabel(r'$log_{10}\tilde{L}_\nu$ [erg/sHz]')
        plt.loglog()
        plt.grid()
        plt.savefig('Figs/Ltildan_m' + str(m) + '_snap' + str(fix))
        plt.show()
    
        plt.figure()
        plt.plot(n_arr, n_arr * lum_tilde_n)
        plt.xlabel(r'$log_{10}\nu$ [Hz]')
        plt.ylabel(r'$log_{10}(\nu\tilde{L}_\nu)$ [erg/s]')
        plt.loglog()
        plt.grid()
        plt.savefig('Figs/n_Ltildan_m' + str(m) + '_snap' + str(fix))
        plt.show()