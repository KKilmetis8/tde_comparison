#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 2023

@author: paola, Konstantinos

Calculate the luminosity that we will use in the blue (BB) curve as sum of multiple BB.

"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

from src.Utilities.isalice import isalice
alice, plot = isalice()

# Vanilla imports
import numpy as np
import matplotlib.pyplot as plt
import h5py
import healpy as hp
from datetime import datetime

# Chocolate Imports
from src.Opacity.LTE_opacity import opacity
from src.Calculators.ray_forest import ray_finder, ray_maker_forest
from src.Luminosity.special_radii_tree import calc_specialr
from src.Calculators.select_observers import select_observer 
from src.Utilities.parser import parse
import src.Utilities.prelude as c
import src.Utilities.selectors as s

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
    const = 2*c.h/c.c**2
    fun = const * n**3 / (np.exp(min(300,c.h*n/(c.Kb*Temperature))) - 1) # Elad: min to avoid overflow
    return fun                                                           # Paola: It doesn't change anything.

def luminosity_n(Temperature: float, Density: float, tau: float, volume: float, n: float):
    """ Luminosity in a cell: L_n = \epsilon e^(-\tau) B_n / B 
    where  B = \sigma T^4/\pi"""
    Tmax = np.exp(17.87) # 5.77e7 K
    Tmin = np.exp(8.666) # 5.80e3 K
    rhomin = np.exp(-44.5)

    if Temperature > Tmax:
        if Density < rhomin:
            Density = rhomin
        # Scale as Kramers the last point 
        kplanck_0 = opacity(Tmax, Density, 'planck')
        k_planck = kplanck_0 * (Temperature/Tmax)**(-3.5)

    elif Temperature < Tmin:
        # NOTE: This is bad, DO IT BETTER 
        # The reason this is bad: there is low T material outside and this extrapolation
        # is not trustworthy so far out
        k_planck = opacity(Tmin, Density, 'planck')

    elif Density < rhomin:
        k_planck = opacity(Temperature, rhomin, 'planck')

    else:
        k_planck = opacity(Temperature, Density, 'planck')

    L = 4  * np.pi * k_planck * volume * np.exp(-min(30,tau)) * planck(Temperature, n)
    return L

def normalisation(L_x: np.array, x_array: np.array, luminosity_fld: float) -> float:
    """ Given the array of luminosity L_x computed over n_array = 10^{x_array} (!!!), 
    find the normalisation constant for L_tilde_n from FLD model. """  
    xLx =  10**(x_array) * L_x
    L = np.trapz(xLx, x_array) 
    L *= np.log(10)
    norm = luminosity_fld / L
    return norm

def spectrum(branch_T, branch_den, branch_cumulative_taus, branch_v, radius, volume, bol_fld):
    lum_n = np.zeros(len(x_arr))
    for i in range(len(branch_cumulative_taus)):
        # Temperature, Density and volume: np.array from near to the BH to far away. 
        # Thus we will use negative index in the for loop.
        # tau: np.array from outside to inside.
        reverse_idx = -i -1
        T = branch_T[reverse_idx]
        rho = branch_den[reverse_idx] 
        opt_depth = branch_cumulative_taus[i]
        cell_vol = volume[reverse_idx] 

        for i_freq, n in enumerate(n_arr): 
            lum_n_cell = luminosity_n(T, rho, opt_depth, cell_vol, n)
            lum_n[i_freq] += lum_n_cell

    # Normalise with the spectra of every observer with red curve (FLD)
    const_norm = normalisation(lum_n, x_arr, bol_fld)
    lum_n = lum_n * const_norm

    return lum_n

# Theta averaging
def dot_prod(xyz_grid):
    dot_product = np.dot(xyz_grid, np.transpose(xyz_grid))
    dot_product[dot_product < 0] = 0
    dot_product = dot_product * 4 / 192

    return dot_product

# MAIN
if __name__ == "__main__":
    save = True
    num = 1000
    if alice:
        pre = '/home/s3745597/data1/TDE/'
        args = parse()
        sim = args.name
        mstar = args.mass
        rstar = args.radius
        Mbh = args.blackhole
        fixes = np.arange(args.first, args.last + 1)
        m = 'AEK'
        check = 'MONO AEK'
    else:
        # Choose BH 
        m = 5
        mstar = 0.5
        rstar = 0.47
        check = 'fid'
        fixes, days = s.select_snap(m, mstar, rstar, check)
    opacity_kind = s.select_opacity(m)

    # Choose the observers: theta in [0, pi], phi in [0,2pi]
    wanted_thetas = [np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi, 0] # x, -x, y, -y, z, -z
    wanted_phis = [0, np.pi, np.pi/2, 3*np.pi/2, 0, 0]

    # Choose freq range
    n_min = 2.08e13 # [Hz]
    n_max = 6.25e23 # [Hz]
    n_spacing = 1000 # Elad used 1000, but no difference
    x_arr = log_array(n_min, n_max, n_spacing)
    n_arr = 10**x_arr
    
    # Save frequency range
    if save:
        if alice:
            pre_saving = '/home/s3745597/data1/TDE/tde_comparison/data/blue'
            np.savetxt(f'{pre_saving}{sim}spectra_freq.txt')
        else:
            with open('data/blue/spectrafreq_m'+ str(m) + '.txt', 'w') as f:
                f.write('# exponents x of frequencies: n = 10^x  \n')
                f.write(' '.join(map(str, x_arr)) + '\n') 
                f.close()
    
    now = datetime.now()
    now = now.strftime("%d/%m/%Y %H:%M:%S")

    # Load data for normalsation 
    fld_data = np.loadtxt(f'data/red/red{sim}_lums.txt')
    luminosity_fld_fix = fld_data
    
    for i, fix in enumerate(fixes): 
        snap = fixes[fix]
        bol_fld = luminosity_fld_fix[fix]
        if alice:
            pre = '/home/s3745597/data1/TDE/'
            filename = f'{pre}/{sim}/snap_{fix}/snap_{fix}.h5'
        
        thetas, phis, stops, xyz_grid = ray_finder(filename)
        rays = ray_maker_forest(snap, m, check, thetas, phis, stops, num, 
                                opacity_kind)

        lum_n = []
        # Find volume of cells
        # Radii has num+1 cell just to compute the volume for num cell. Then we delete the last radius cell
        for j in range(len(thetas)):
            print('ray', j)

            branch_indexes = rays.tree_indexes[j]
            branch_T = rays.T[j]
            branch_den = rays.den[j]
            radius = rays.radii[j]
            branch_v = rays.v[j]

            volume = np.zeros(len(radius)-1)
            for i in range(len(volume)): 
                dr = 2*(radius[2]-radius[1]) / (radius[2] + radius[1]) #radius[i+1] - radius[i]  
                volume[i] =  4 * np.pi * radius[i]**3 * dr #4 * np.pi * radius[i]**2 * dr / 192  

            radius = np.delete(radius, -1)

            # Get thermalisation radius
            _, branch_cumulative_taus, _, _, _ = calc_specialr(branch_T, branch_den, radius, branch_indexes, opacity_kind, select = 'thermr')

            # Compute specific luminosity of every observers
            lum_n_ray = spectrum(branch_T, branch_den, branch_cumulative_taus, branch_v, radius, volume, bol_fld)

            lum_n.append(lum_n_ray)
        
        # Theta average
        dot_product = dot_prod(xyz_grid) #dot_prod(xyz_selected, thetas, phis)
        lum_n_selected = np.dot(dot_product, lum_n)

        # Select the observer for single spectrum and compute the dot product
        dirs6 = []
        for idx in range(len(wanted_thetas)):
            wanted_theta = wanted_thetas[idx]
            wanted_phi = wanted_phis[idx]
            wanted_index = select_observer(wanted_theta, wanted_phi, thetas, phis)
            print('index ', wanted_index)
            dirs6.append(lum_n_selected[wanted_index])

        if save:
            if alice:
                pre_saving = '/home/s3745597/data1/TDE/tde_comparison/data/blue/'
                np.savetxt(f'{pre_saving}{sim}_spectrum.txt', lum_n_selected)
                np.savetxt(f'{pre_saving}{sim}_dirs6.txt', dirs6)
            # # Save data and plot
            # if save:
            #     if alice:
            #         pre_saving = '/home/s3745597/data1/TDE/tde_comparison/data/blue'
            #     else:
            #         pre_saving = 'data/blue/'
            #     with open(f'{pre_saving}{sim}spectrum.txt', 'a') as fselect:
            #         fselect.write(f'#snap {snap} L_tilde_n (theta, phi) = ({np.round(wanted_theta,4)},{np.round(wanted_phi,4)}) with num = {num} \n')
            #         fselect.write(' '.join(map(str, lum_n_selected[wanted_index])) + '\n')
            #         fselect.close()
                
