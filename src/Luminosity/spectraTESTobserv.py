#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 2023

@author: paola 

Calculate the luminosity that we will use in the blue (BB) curve as sum of multiple BB.

"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

from src.Utilities.isalice import isalice
alice, plot = isalice()

# Vanilla imports
import numpy as np
import matplotlib.pyplot as plt

# Chocolate Imports
from src.Opacity.old_opacity import old_opacity #TEST OLD OPACITY
from src.Calculators.ray_tree import ray_maker
from src.Luminosity.special_radii_tree import get_specialr
from src.Luminosity.select_path import select_snap
from datetime import datetime
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [5 , 4]

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

def select_observer(wanted_theta, wanted_phi, thetas, phis):
    """ Gives thetas, phis from helpix and 
    the index of the points closer to the one given by (wanted_theta, wanted_phi)"""
    dist = np.zeros(192)
    list_index = []
    for i in range(len(thetas)): 
        delta_theta = wanted_theta -  thetas[i]
        delta_phi = wanted_phi -  phis[i]
        # Haversine formula
        arg = np.sin(delta_theta / 2)**2 + np.cos(wanted_theta) * np.cos(thetas[i]) * np.sin(delta_phi/2)**2
        dist[i] = 2 * np.arctan2( np.sqrt(arg), np.sqrt(1-arg))
        if np.abs(np.abs(dist[i])-dist.min())<1e-16:
            print(i, dist[i])
            list_index.append(i)

    return list_index


def planck(Temperature: float, n: float) -> float:
    """ Planck function in a cell. It needs temperature and frequency n. """
    const = 2*h/c**2
    fun = const * n**3 / (np.exp(min(300,h*n/(Kb*Temperature))) - 1) #min to avoid overflow

    return fun

def luminosity_n(Temperature: float, Density: float, tau: float, volume: float, n: float):
    """ Luminosity in a cell: L_n = \epsilon e^(-\tau) B_n / B 
    where  B = \sigma T^4/\pi"""
    Tmax = np.power(10,8) #np.exp(17.87) #TEST OLD OPACITY
    if Temperature > Tmax:
        # Scale as Kramers the last point 
        # kplanck_0 = opacity(Tmax, Density, 'planck', ln = False)
        kplanck_0 = old_opacity(Tmax, Density, 'planck') #TEST OLD OPACITY
        k_planck = kplanck_0 * (Temperature/Tmax)**(-3.5)
    else:
        # k_planck = opacity(Temperature, Density, 'planck', ln = False)
        k_planck = old_opacity(Temperature, Density, 'planck') #TEST OLD OPACITY

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

def find_sph_coord(theta,phi):
    x = np.sin(np.pi-theta) * np.cos(phi) #because theta should start from the z axis: we're flipped
    y = np.sin(np.pi-theta) * np.sin(phi)
    z = np.cos(np.pi-theta)
    xyz = [x, y, z]
    return xyz

def normalisation(L_x: np.array, x_array: np.array, luminosity_fld: float) -> float:
    """ Given the array of luminosity L_x computed over n_array = 10^{x_array} (!!!), 
    find the normalisation constant for L_tilde_n from FLD model. """  
    xLx =  10**(x_array) * L_x
    L = np.trapz(xLx, x_array) 
    L *= np.log(10)
    norm = luminosity_fld / L
    return norm

def spectrum(rays_T, rays_den, rays_cumulative_taus, volume, bol_fld):
    lum_n = np.zeros((len(rays_T), len(x_arr)))

    for j in range(len(rays_T)):
        print('ray :', j)

        for i in range(len(rays_cumulative_taus[j])):        
            # Temperature, Density and volume: np.array from near to the BH to far away. 
            # Thus we will use negative index in the for loop.
            # tau: np.array from outside to inside.
            reverse_idx = -i -1
            T = rays_T[j][reverse_idx]
            rho = rays_den[j][reverse_idx] 
            opt_depth = rays_cumulative_taus[j][i]
            cell_vol = volume[reverse_idx] #rays_vol[j][reverse_idx] * Rsol_to_cm**3  if you use simulation volume

            for i_freq, n in enumerate(n_arr): #we need linearspace
                lum_n_cell = luminosity_n(T, rho, opt_depth, cell_vol, n)
                #lum_n_cell *= dot_product[j]
                lum_n[j][i_freq] += lum_n_cell

        # Normalise with the spectra of every observer with red curve (FLD)
        const_norm = normalisation(lum_n[j], x_arr, bol_fld)
        lum_n[j] = lum_n[j] * const_norm

    return lum_n

def dot_prod(xyz_selected, thetas, phis):
    dot_product = np.zeros(len(thetas))
    for iobs in range(len(thetas)):
        xyz = find_sph_coord(thetas[iobs], phis[iobs])
        dot_product[iobs] = np.dot(xyz_selected, xyz)
        # set the negative dot product to 0
        if dot_product[iobs] < 0:
            dot_product[iobs] = 0

    dot_product *= 4/192 # normalisation from Elad
    return dot_product

# MAIN
if __name__ == "__main__":
    save = True

    # Choose BH 
    m = 6
    check = 'fid'
    num = 1000
    snapshots, days = select_snap(m, check)

    # Choose the observers: theta in [0, pi], phi in [0,2pi]
    wanted_theta = np.pi/2
    wanted_phi = 0

    # Choose freq range
    n_min = 2.08e13
    n_max = 6.25e23
    n_spacing = 100 # Elad used 1000, but no difference
    x_arr = log_array(n_min, n_max, n_spacing)
    n_arr = 10**x_arr
    
    # Save frequency range
    if save:
        if alice:
            pre_saving = '/home/s3745597/data1/TDE/tde_comparison/data/'
            with open(f'{pre_saving}spectrafreq_m{m}.txt', 'w') as f:
                f.write('# exponents x of frequencies: n = 10^x  \n')
                f.write(' '.join(map(str, x_arr)) + '\n') 
                f.close()
        else:
            with open('data/blue/spectrafreq_m'+ str(m) + '.txt', 'w') as f:
                f.write('# exponents x of frequencies: n = 10^x  \n')
                f.write(' '.join(map(str, x_arr)) + '\n') 
                f.close()
    
    now = datetime.now()
    now = now.strftime("%d/%m/%Y %H:%M:%S")

    # Load data for normalsation 
    fld_data = np.loadtxt('data/red/reddata_m'+ str(m) + check +'.txt')
    luminosity_fld_fix = fld_data[1]
    
    for idx_sn in range(1,2): #so you take snap 881
        snap = snapshots[idx_sn]
        bol_fld = luminosity_fld_fix[idx_sn]
        print(f'Snap {snap}')

        # Find observers 
        tree_indexes, observers, rays_T, rays_den, _, radii, rays_vol = ray_maker(snap, m, check, num)
        # Find voulume of cells
        # Radii has num+1 cell just to compute the volume for num cell. Then we delete the last radius cell
        volume = np.zeros(len(radii)-1)
        for i in range(len(volume)): 
            dr = radii[i+1] - radii[i]  
            volume[i] =  4 * np.pi * radii[i]**2 * dr / 192   

        radii = np.delete(radii, -1)
        # Select the observer for single spectrum and compute the dot product

        thetas = np.zeros(192)
        phis = np.zeros(192) 
        for iobs in range(len(observers)): 
            thetas[iobs] = observers[iobs][0]
            phis[iobs] =  observers[iobs][1]

        wanted_indexes = select_observer(wanted_theta, wanted_phi, thetas, phis)
        
        # Get thermalisation radius
        _, rays_cumulative_taus, _, _, _ = get_specialr(rays_T, rays_den, radii, tree_indexes, select = 'thermr')
        # Compute specific luminosity of every observers
        lum_n = spectrum(rays_T, rays_den, rays_cumulative_taus, volume, bol_fld)

        for idx in range(len(wanted_indexes)):
            wanted_index = wanted_indexes[idx]
            xyz_selected = find_sph_coord(thetas[wanted_index], phis[wanted_index])
            dot_product = dot_prod(xyz_selected, thetas, phis)

            lum_n_selected = np.zeros(len(x_arr))
            for j in range(len(lum_n)):
                if dot_product[j] == 0:
                    continue
                for i in range(len(x_arr)):
                    lum_n_single = lum_n[j][i] * dot_product[j]
                    lum_n_selected[i] += lum_n_single

            # Save data and plot
            if save:
                if alice:
                    pre_saving = '/home/s3745597/data1/TDE/tde_comparison/data/'
                else:
                    pre_saving = 'data/blue/'
            with open(f'{pre_saving}TESTobserv_nLn_single_m{m}_{snap}.txt', 'a') as fselect:
                fselect.write(f'#snap {snap} L_tilde_n (theta, phi) = ({np.round(wanted_theta,4)},{np.round(wanted_phi,4)}) with helpix_idx = {wanted_index} \n')
                fselect.write(' '.join(map(str, lum_n_selected)) + '\n')
                fselect.close()
                