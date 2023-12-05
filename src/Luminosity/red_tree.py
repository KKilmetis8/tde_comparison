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
import math
from scipy.spatial import KDTree
from datetime import datetime
# Custom Imports
from src.Opacity.opacity_table import opacity
from src.Calculators.ray_tree import ray_maker, isalice
alice = isalice()
from src.Luminosity.special_radii_tree import get_photosphere
from astropy.coordinates import spherical_to_cartesian, cartesian_to_spherical
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [5 , 4]

#%% Constants & Converter
G = 6.6743e-11 # SI
Msol = 1.98847e30 # kg
Rsol = 6.957e8 # m
t = np.sqrt(Rsol**3 / (Msol*G )) # Follows from G = 1
c = 3e8 * t/Rsol # simulator units. Need these for the PW potential
c_cgs = 3e10 # [cm/s]
Msol_to_g = 1.989e33 # [g]
Rsol_to_cm = 6.957e10 # [cm]
den_converter = Msol_to_g / Rsol_to_cm**3
en_den_converter = Msol_to_g / (Rsol_to_cm  * t**2)
pre = '/home/s3745597/data1/TDE/tde_comparison'
#%%
##
# FUNCTIONS
##
###
def select_fix(m, check = 'fid'):
    if alice:
        if m == 6 and check == 'fid':
            snapshots = np.arange(844, 1008 + 1, step = 1)
        if m == 4 and check == 'fid':
            snapshots = np.arange(100, 344 + 1)
        if m == 4 and check == 'S60ComptonHiRes':
            SNAPSHOTS = np.arange(210, 271 + 1)
    else:
        if m == 4:
            snapshots = [233] #, 254, 263, 277 , 293, 308, 322]
            days = [1]# , 1.2, 1.3, 1.4, 1.56, 1.7, 1.8] 
        if m == 6:
            snapshots = [844, 881, 925, 950]# 1008] 
            days = [1, 1.1, 1.3, 1.4]# 1.608] 
    return snapshots, days

def find_neighbours(fix, m, check, tree_index_photo, dist_neigh):
    """
     For every ray, find the cells that are at +- fixed distance from photosphere.
     fixed distance = 2 * dimension of simulation cell at the photosphere

     Parameters
     ----------
     fix: int.
           Snapshot number.
     m: int.
        Exponent of BH mass
     tree_index_photo: 1D array.
                 Photosphere index in the tree.
     dist_neigh : 1D array.
                2(3πV/4)^(1/3), Distance from photosphere (in Rsol).

     Returns
     -------
     grad_E: array.
             Energy gradient for every ray at photosphere (CGS). 
     energy_high: array.
             Energy for every ray in a cell outside photosphere (CGS). 
     T_high: array.
             Temperature for every ray in a cell outside photosphere (CGS). 
     den_high: array.
             Density for every ray in a cell outside photosphere (CGS). 

    """
    Mbh = 10**m 
    Rt =  Mbh**(1/3) # Msol = 1, Rsol = 1
    fix = str(fix)
    sim = str(m) + '-' + check
    if alice:
        pre = '/home/s3745597/data1/TDE/'
        # Import
        X = np.load(pre + sim + '/snap_'  + fix + '/CMx_' + fix + '.npy') - Rt
        Y = np.load(pre + sim + '/snap_'  + fix + '/CMy_' + fix + '.npy')
        Z = np.load(pre + sim + '/snap_'  + fix + '/CMz_' + fix + '.npy')
        T = np.load(pre + sim + '/snap_'  + fix + '/T_' + fix + '.npy')
        Den = np.load(pre + sim + '/snap_'  + fix + '/Den_' + fix + '.npy')
        Rad = np.load(pre + sim + '/snap_'  +fix + '/Rad_' + fix + '.npy')
    else:
        X = np.load( str(m) + '/'  + str(fix) + '/CMx_' + str(fix) + '.npy') - Rt
        Y = np.load( str(m) + '/'  + str(fix) + '/CMy_' + str(fix) + '.npy')
        Z = np.load( str(m) + '/'  + str(fix) + '/CMz_' + str(fix) + '.npy')
        Rad = np.load(str(m) + '/'  +str(fix) + '/Rad_' + str(fix) + '.npy')
        T = np.load( str(m) + '/'  + str(fix) + '/T_' + str(fix) + '.npy')
        Den = np.load( str(m) + '/'  + str(fix) + '/Den_' + str(fix) + '.npy')

    # convert in CGS
    Rad *= Den 
    Rad *= en_den_converter

    # make a tree
    sim_value = [X, Y, Z] 
    sim_value = np.transpose(sim_value) # shape (number_points, 3). Need it for the tree.
    sim_tree = KDTree(sim_value)

    # store data of R_{ph} (lenght in Rsol)
    tree_index_photo = [int(x) for x in tree_index_photo]
    xyz_obs = [X[tree_index_photo], Y[tree_index_photo], Z[tree_index_photo]]
    r_obs, theta_obs, phi_obs = cartesian_to_spherical(xyz_obs[0], xyz_obs[1],
                                                       xyz_obs[2])
    # Find the coordinates of neighborus.
    # For every ray, they have its (theta,phi) and R = R_{ph} +- 2dr
    r_low = r_obs - dist_neigh
    r_high = r_obs + dist_neigh

    # convert to cartesian and query 
    x_low, y_low, z_low  = spherical_to_cartesian(r_low, theta_obs, phi_obs)
    x_high, y_high, z_high  = spherical_to_cartesian(r_high, theta_obs, phi_obs)
    idx_low = np.zeros(len(tree_index_photo))
    idx_high = np.zeros(len(tree_index_photo))
    grad_r = np.zeros(len(tree_index_photo))
    grad_xyz = []
    magnitude = np.zeros(len(tree_index_photo))
   
    # Find inner and outer neighbours
    for i in range(len(x_low)):
        _, idx_l = sim_tree.query([x_low[i], y_low[i], z_low[i]])
        _, idx_h = sim_tree.query([x_high[i], y_high[i], z_high[i]])
        idx_low[i] = idx_l
        idx_high[i] = idx_h
        xyz_low = np.array([X[idx_l], Y[idx_l], Z[idx_l]])
        xyz_high = np.array([X[idx_h], Y[idx_h], Z[idx_h]])
        
        # Diff is vector
        diff = 1 / np.subtract(xyz_high, xyz_low)
        diff = diff / Rsol_to_cm # convert to CGS
        
        # Unitary vector in the r direction, for each observer
        rhat = [np.sin(theta_obs[i]) * np.cos(phi_obs[i]),
                np.sin(theta_obs[i]) * np.sin(phi_obs[i]),
                np.cos(theta_obs[i])
                ]
        grad_r[i] = np.dot(diff, rhat) # Project
        magnitude[i] = 1 / (np.linalg.norm(np.subtract(xyz_high, xyz_low)) * Rsol_to_cm) 
    
    # store data of neighbours
    idx_low = [int(x) for x in idx_low] #necavoid dumb stuff with indexing later
    idx_high = [int(x) for x in idx_high] #same 
    energy_low = Rad[idx_low]
    energy_high = Rad[idx_high]
    T_high = T[idx_high]
    den_high = Den[idx_high]

    # compute the gradient 
    deltaE = energy_high - energy_low

    #grad_xyz = grad_xyz / Rsol_to_cm
    grad_Er = deltaE * grad_r
    magnitude *= np.abs(deltaE)
    
    return grad_Er, magnitude, energy_high, T_high, den_high

    
def flux_calculator(grad_E, magnitude, selected_energy, 
                    selected_temperature, selected_density):
    """
    Get the flux for every observer.

    Parameters
    ----------
    grad_E: array.
            Energy gradient for every ray at photosphere. 
    selected_energy: array.
            Energy for every ray in a cell outside photosphere. 
    selected_temperature: array.
            Temperature for every ray in a cell outside photosphere. 
    selected_density: array.
            Density for every ray in a cell outside photosphere. 
        
    Returns
    -------
    f: array
        Flux at every ray.
    """
    f = np.zeros(len(grad_E))
    max_count = 0
    max_but_zero_count = 0
    zero_count = 0
    flux_zero = 0
    flux_count = 0

    for i in range(len(selected_energy)):
        Energy = selected_energy[i]
        max_travel = np.sign(-grad_E[i]) * c_cgs * Energy 
        
        Temperature = selected_temperature[i]
        Density = selected_density[i]

        # Ensure we can interpolate
        rho_low = np.exp(-45)
        T_low = np.exp(8.77)
        T_high = np.exp(17.876)
        
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
        
        # T too high => scattering
        if Temperature > T_high:
            Tscatter = np.exp(17.87)
            k_ross = opacity(Tscatter, Density, 'scattering', ln = False)
        else:    
            # Get Opacity, NOTE: Breaks Numba
            k_ross = opacity(Temperature, Density, 'rosseland', ln = False)
        
        # Calc R, eq. 28
        R_kr = magnitude[i] /  (k_ross * Energy)
        invR = 1 / R_kr
        R_kr = float(R_kr) # to avoid dumb thing with tanh(R)
    
        # Calc lambda, eq. 27
        coth = 1 / np.tanh(R_kr)
        lamda = invR * (coth - invR)
        # Calc Flux, eq. 26
        Flux =  - c_cgs * grad_E[i]  * lamda / k_ross
        
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
    print('Zero due to: \n- max travel: ', max_but_zero_count)
    print('- T_low:', zero_count)
    print('- flux:', flux_zero) 
    print('Flux: ', flux_count) 
    return f

def doer_of_thing(fix, m, check, num = 1000):
    """
    Gives bolometric L 
    """
    tree_indexes, rays_T, rays_den, rays, radii, rays_vol = ray_maker(fix, m, check, num)
    _, _, rays_photo, rays_index_photo, tree_index_photo = get_photosphere(rays_T, rays_den, radii, tree_indexes)
    
    dim_ph = np.zeros(len(rays_index_photo))
    for j in range(len(rays_index_photo)):
        find_index_cell = int(rays_index_photo[j])
        vol_ph = rays_vol[j][find_index_cell]
        dim_ph[j] = (3 * vol_ph /(4 * np.pi))**(1/3) #in solar units
    dist_neigh = 2 * dim_ph
    # dist_neigh *= Rsol_to_cm #convert in CGS

    # Find the cell outside the photosphere and save its quantities
    # grad_E, energy_high, T_high, den_high  = find_neighbours(rays_T, rays_den, rays, radii, 
    #                                                          rays_index_photo, dist_neigh)
    grad_E, magnitude, energy_high, T_high, den_high  = find_neighbours(fix, m, tree_index_photo, dist_neigh)

    # Calculate Flux and see how it looks
    flux = flux_calculator(grad_E, magnitude, energy_high, 
                           T_high, den_high)

    # Calculate luminosity 
    lum = np.zeros(len(flux))
    zero_count = 0
    neg_count = 0
    for i in range(len(flux)):
        # Turn to luminosity
        if flux[i] == 0:
            zero_count += 1
        if flux[i] < 0:
            neg_count += 1
            flux[i] = 0 

        lum[i] = flux[i] * 4 * np.pi * rays_photo[i]**2

    # Average in observers
    lum = np.sum(lum)/192

    print('Tot zeros:', zero_count)
    print('Negative: ', neg_count)      
    print('Fix %i' %fix, ', Lum %.3e' %lum, '\n---------' )
    return lum
#%%
##
# MAIN
##
if __name__ == "__main__":
    save = True
    plot = False
    m = 4 # Choose BH
    check = 'S60ComptonHires' # Choose check S60ComptonHires
    sim = str(m) + '-' + check
    fixes, days = select_fix(m, check)
    lums = []
            
    for idx in range(0,len(fixes)):
        lum = doer_of_thing(fixes[idx], m, check)
        lums.append(lum)
    
    if save:
        if alice:
            pre = '/home/s3745597/data1/TDE/'
            np.savetxt('red_backup_save'+ sim + '.txt', lums)
            np.savetxt(pre + 'tde_comparison/data/alicered'+ sim + '.txt', lums)
        else:
             with open('data/red/new_reddata_m'+ str(m) + '.txt', 'a') as flum:
                 flum.write('# t/t_fb\n') 
                 flum.write(' '.join(map(str, days)) + '\n')
                 flum.write('# Lum \n') 
                 flum.write(' '.join(map(str, lums)) + '\n')
                 flum.close() 

    #%% Plotting
    if plot:
        plt.figure()
        plt.plot(lums, '-o', color = 'maroon')
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
        plt.savefig('Final plot/ourred' + str(m) + '.png')
        plt.show()

