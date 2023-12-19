#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 9 2023

@author: konstantinos, paola 

Calculate the luminosity that we will use in the blue (BB) curve as sum of multiple BB.

"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

from src.Utilities.isalice import isalice
alice, plot = isalice()

# Vanilla imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# Chocolate Imports
from src.Opacity.opacity_table import opacity
from src.Calculators.ray_tree import ray_maker
from src.Luminosity.special_radii_tree import get_specialr
from src.Calculators.select_observers import select_observer 
from src.Luminosity.select_path import select_snap
from datetime import datetime
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [5 , 4]
pre = '/home/s3745597/data1/TDE/tde_comparison'

# Constants
c = 2.99792458e10 #[cm/s]
h = 6.62607015e-27 #[gcm^2/s]
Kb = 1.380649e-16 #[gcm^2/s^2K]
alpha = 7.5646 * 10**(-15) # radiation density [erg/cm^3K^4]
Rsol_to_cm = 6.957e10
# rhat = [np.sin(theta_obs[i]) * np.cos(phi_obs[i]),
#                 np.sin(theta_obs[i]) * np.sin(phi_obs[i]),
#                 np.cos(theta_obs[i])
#                 ]
#%%
###
# FUNCTIONS
###

def log_array(n_min, n_max, lenght):
    x_min = np.log10(n_min)
    x_max = np.log10(n_max)
    x_arr = np.linspace(x_min, x_max , num = lenght)
    return x_arr

def planck(Temperature: float, n: float) -> float:
    """ Planck function in a cell. It needs temperature and frequency n. """
    const = 2*h/c**2
    fun = const * n**3 / (np.exp(h*n/(Kb*Temperature))-1)

    return fun

def luminosity_n(Temperature: float, Density: float, tau: float, volume: float, n: float):
    """ Luminosity in a cell: L_n = \epsilon e^(-\tau) B_n / B 
    where  B = \sigma T^4/\pi"""
    T_high = np.exp(17.87)
    if Temperature > T_high:
        Tmax = np.exp(17.87)
        # Scale as Kramers the last point 
        kplank_0 = opacity(Tmax, rho, 'planck', ln = False)
        k_planck = kplank_0 * (T/Tmax)**(-3.5)
    else:
        k_planck = opacity(Temperature, Density, 'planck', ln = False)

    L = 4  * np.pi * k_planck * volume * np.exp(-tau) * planck(Temperature, n)
    return L

def normalisation(L_x: np.array, x_array: np.array, luminosity_fld: float) -> float:
    """ Given the array of luminosity L_x computed over n_array = 10^{x_array} (!!!), 
    find the normalisation constant for L_tilde_n from FLD model. """  
    xLx =  10**(x_array) * L_x
    L = np.trapz(xLx, x_array) 
    L *= np.log(10)
    norm = luminosity_fld / L
    return norm

def select_rays(wanted_theta, wanted_phi, rays_T, rays_den, rays_cumulative_taus):
    _, _, wanted_indexes = select_observer(wanted_theta, wanted_phi)
    rays_T_new = []
    rays_den_new = []
    rays_cumulative_taus_new = []
    for idx in wanted_indexes:
        rays_T_new.append(rays_T[idx])
        rays_den_new.append(rays_den[idx])
        rays_cumulative_taus_new.append(rays_cumulative_taus[idx])

    return rays_T_new, rays_den_new, rays_cumulative_taus_new

# MAIN
if __name__ == "__main__":
    plot = True
    save = True
    snap = 881

    # Choose BH 
    m = 6
    check = 'fid'
    num = 1000

    # Choose the observers
    wanted_theta = 0
    wanted_phi = np.pi/2

    # Choose freq range
    n_min = 2.08e13
    n_max = 6.25e23
    n_spacing = 100
    x_arr = log_array(n_min, n_max, n_spacing)
    
    # Save frequency range
    if save:
        with open('data/blue/spectrafreq_m'+ str(m) + '.txt', 'w') as f:
            f.write('# exponents x of frequencies: n = 10^x  \n')
            f.write(' '.join(map(str, x_arr)) + '\n') 
            f.close()
    
    now = datetime.now()
    now = now.strftime("%d/%m/%Y %H:%M:%S")
    n_arr = 10**x_arr

    snapshots, days = select_snap(m, check)
    
    #%% Get thermalisation radius
    tree_indexes, observers, rays_T, rays_den, _, radii, _ = ray_maker(snap, m, check, num)

    _, _, wanted_index = select_observer(wanted_theta, wanted_phi, observers)
    x_selected = np.sin(observers[wanted_index][0]) * np.cos(observers[wanted_index][1])
    y_selected = np.sin(observers[wanted_index][0]) * np.sin(observers[wanted_index][1])
    z_selected = np.cos(observers[wanted_index][0])
    xyz_selected = [x_selected, y_selected, z_selected]
    cross_dot = np.zeros(len(observers))
    for iobs in range(len(observers)):
        x = np.sin(observers[iobs][0]) * np.cos(observers[iobs][1])
        y = np.sin(observers[iobs][0]) * np.sin(observers[iobs][1])
        z = np.cos(observers[iobs][0])
        xyz = [x,y,z]
        cross_dot[iobs] = np.dot(xyz_selected, xyz)
        if cross_dot[iobs] < 0:
            cross_dot[iobs] = 0

    _, rays_cumulative_taus, _, _, _ = get_specialr(rays_T, rays_den, radii, tree_indexes, select = 'thermr')

    #%%   
    volume = np.zeros(len(radii))
    for i in range(len(radii)-1): 
        dr = radii[i+1] - radii[i]
        volume[i] = 4 * np.pi * radii[i]**2 * dr / 192         

    lum_n = np.zeros(len(x_arr))

    for j in range(len(rays_T)):
        print('ray :', j)
        for i in range(len(rays_cumulative_taus[j])):        
            # Temperature, Density and volume: np.array from near to the BH
            # to far away. 
            # Thus we will use negative index in the for loop.
            # tau: np.array from outside to inside.
            reverse_idx = -i -1
            T = rays_T[j][reverse_idx]
            rho = rays_den[j][reverse_idx] 
            opt_depth = rays_cumulative_taus[j][i]
            cell_vol = volume[reverse_idx]
            
            # Ensure we can interpolate
            T_low = np.exp(8.666)
            T_high = np.exp(17.87)
            rho_low = np.exp(-49.2)
            
            # Out of table
            if rho < rho_low:
                print('rho low')
                continue
            
            # Opaque
            if T < T_low:
                print('T low')
                #continue    
                T = np.exp(8.7)
            
            for i, n in enumerate(n_arr): #we need linearspace
                lum_n_cell = luminosity_n(T, rho, opt_depth, cell_vol, n)
                lum_n_cell *= cross_dot[j]
                lum_n[i] += lum_n_cell

    const_norm = 1/192
    lum_tilde_n = lum_n * const_norm

    # Save data and plot
    if save:
        with open(f'data/blue/nLn_single_m{m}.txt', 'a') as fselect:
            fselect.write(f'#snap {snap} L_tilde_n (theta, phi) = ({np.round(wanted_theta,4)},{np.round(wanted_phi,4)}) \n')
            fselect.write(' '.join(map(str, lum_tilde_n)) + '\n')
            fselect.close()
  
    if plot:
        fig, ax1 = plt.subplots( figsize = (6,6) )
        ax1.plot(n_arr, n_arr * lum_tilde_n)
        ax2 = ax1.twiny()
        ax1.set_xlabel(r'$log_{10}\nu$ [Hz]')
        ax1.set_ylabel(r'$log_{10}(\nu\tilde{L}_\nu)$ [erg/s]')
        ax1.set_ylim(1e39, 2e44)
        ax1.loglog()
        ax1.grid()
        wavelength = np.divide(c, n_arr) * 1e8 # A
        ax2.plot(wavelength, n_arr * lum_tilde_n)
        ax2.invert_xaxis()
        ax2.loglog()
        ax2.set_xlabel(r'Wavelength [\AA]')
        # plt.savefig(f'Figs/n_Ltildan_m{m}_snap{snap}')
        plt.show()
                    