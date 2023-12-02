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
from src.Calculators.ray_tree import ray_maker
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
alice = False
#%%
##
# FUNCTIONS
##
###
def select_fix(m):
    if m == 4:
        snapshots = [233] #, 254, 263, 277 , 293, 308, 322]
        days = [1]# , 1.2, 1.3, 1.4, 1.56, 1.7, 1.8] 
    if m == 6:
        if alice:
            snapshots = np.arange(844, 1008 + 1)
            days = [1.00325,1.007,1.01075,1.01425,1.018,1.02175,1.0255,1.029,1.03275,1.0365,1.04025,1.04375,1.0475,1.05125,1.055,1.0585,1.06225,1.066,1.06975,1.07325,1.077,1.08075,1.0845,1.088,1.09175,1.0955,1.09925,1.10275,1.1065,1.11025,1.114,1.1175,1.12125,1.125,1.12875,1.13225,1.136,1.13975,1.1435,1.147,1.15075,1.1545,1.15825,1.16175,1.1655,1.16925,1.173,1.1765,1.18025,1.184,1.18775,1.19125,1.195,1.19875,1.2025,1.206,1.20975,1.2135,1.21725,1.22075,1.2245,1.22825,1.232,1.2355,1.23925,1.243,1.24675,1.25025,1.254,1.25775,1.2615,1.265,1.26875,1.2725,1.27625,1.27975,1.2835,1.28725,1.291,1.2945,1.29825,1.302,1.30575,1.30925,1.313,1.31675,1.3205,1.324,1.32775,1.3315,1.33525,1.33875,1.3425,1.34625,1.35,1.3535,1.35725,1.361,1.36475,1.36825,1.372,1.37575,1.3795,1.383,1.38675,1.3905,1.39425,1.39775,1.4015,1.40525,1.409,1.4125,1.41625,1.42,1.42375,1.42725,1.431,1.43475,1.4385,1.442,1.44575,1.4495,1.45325,1.45675,1.4605,1.46425,1.468,1.4715,1.47525,1.479,1.48275,1.48625,1.49,1.49375,1.4975,1.501,1.50475,1.5085,1.51225,1.51575,1.5195,1.52325,1.527,1.5305,1.53425,1.538,1.54175,1.54525,1.549,1.55275,1.5565,1.56,1.56375,1.5675,1.57125,1.57475,1.5785,1.58225,1.586,1.5895,1.59325,1.597,1.60075,1.60425,1.608]
        else:
            snapshots = [844, 881, 925, 950, 980]#, 1008] 
            days = [1, 1.1, 1.3, 1.4, 1.5]#, 1.608] 
    return snapshots, days

def find_neighbours(fix, m, tree_index_photo, dist_neigh):
    X = np.load( str(m) + '/'  + str(fix) + '/CMx_' + str(fix) + '.npy')
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
    sim_value = np.transpose(sim_value) # array of dim (number_points, 3)
    sim_tree = KDTree(sim_value)

    # store data of the points to query in solar unit
    tree_index_photo = [int(x) for x in tree_index_photo]
    xyz_selected = [X[tree_index_photo], Y[tree_index_photo], Z[tree_index_photo]]
    r_selected, theta_selected, phi_selected = cartesian_to_spherical(xyz_selected[0], 
                                                                      xyz_selected[1], xyz_selected[2])
    r_neigh_low = r_selected - dist_neigh
    r_neigh_high = r_selected + dist_neigh
    delta_r = r_neigh_high - r_neigh_low
    delta_r *= Rsol_to_cm # convert to CGS for the gradient

    x_low, y_low, z_low  = spherical_to_cartesian(r_neigh_low, theta_selected, phi_selected)
    x_high, y_high, z_high  = spherical_to_cartesian(r_neigh_high, theta_selected, phi_selected)

    idx_low = np.zeros(len(tree_index_photo))
    idx_high = np.zeros(len(tree_index_photo))
    for i in range(len(x_low)):
        _, idx_l = sim_tree.query([x_low[i], y_low[i], z_low[i]])
        _, idx_h = sim_tree.query([x_high[i], y_high[i], z_high[i]])
        idx_low[i] = idx_l
        idx_high[i] = idx_h

    idx_low = [int(x) for x in idx_low]
    idx_high = [int(x) for x in idx_high]
    energy_neigh_low = Rad[idx_low]
    energy_neigh_high = Rad[idx_high]
    T_neigh_high = T[idx_high]
    den_neigh_high = Den[idx_high]

    deltaE = energy_neigh_high - energy_neigh_low
    grad_E = deltaE / delta_r
    
    return grad_E, energy_neigh_high, T_neigh_high, den_neigh_high

# def find_neighbours(rays_T, rays_den, rays, radii, rays_index_photo, dist_neigh):
#     """
#     For every ray, find the cells that are at +- fixed distance from photosphere.
#     fixed distance = dimension of simulation cell at the photosphere

#     Parameters
#     ----------
#     rays_T: n-D arrays.
#             Temperature of every ray/cell (CGS).
#     rays_den: n-D arrays.
#             Density of every ray/cell (CGS).
#     rays: n-D arrays.
#         Energy of every ray/cell (CGS).
#     radii: 1D array.
#             Radius (CGS).
#     rays_index_photo: 1D array.
#                 Photosphere index in our rays.
#     dist_neigh : 1D array.
#               Distance from photosphere (CGS).

#     Returns
#     -------
#     grad_E: array.
#             Energy gradient for every ray at photosphere. 
#     energy_high: array.
#             Energy for every ray in a cell outside photosphere. 
#     T_high: array.
#             Temperature for every ray in a cell outside photosphere. 
#     den_high: array.
#             Density for every ray in a cell outside photosphere. 
#     """
#     # convert the elements in rays_index_photo in int
#     rays_index_photo = rays_index_photo.astype(int) 
#     dist_neigh_high = radii[rays_index_photo] + dist_neigh
#     dist_neigh_low = radii[rays_index_photo] - dist_neigh

#     grad_E = np.zeros(len(rays_index_photo))
#     energy_high = np.zeros(len(rays_index_photo))
#     T_high = np.zeros(len(rays_index_photo))
#     den_high = np.zeros(len(rays_index_photo))

#     for i in range(len(rays_index_photo)):
 
#         # Isolate each ray
#         T_of_single_ray = rays_T[i]
#         Den_of_single_ray = rays_den[i]
#         energy_of_single_ray = rays[i]

#         # Find and store the neighbour with R ~ R_{ph} + dist_neigh
#         index_high = np.argmin((np.abs(dist_neigh_high[i] - radii)))
#         radius_high = radii[index_high]
#         energy_high[i] = energy_of_single_ray[index_high]
#         T_high[i] = T_of_single_ray[index_high]
#         den_high[i] = Den_of_single_ray[index_high]

#         # Find the neighbour with R ~ R_{ph} - dist_neigh
#         index_low = np.argmin((np.abs(dist_neigh_low[i] - radii)))
#         radius_low = radii[index_low]
#         energy_low = energy_of_single_ray[index_low]

#         # Compute the gradient 
#         dr = radius_high - radius_low
#         grad_E[i] = (energy_high[i] - energy_low) / dr

#     return grad_E, energy_high, T_high, den_high

    
def flux_calculator(grad_E, selected_energy, 
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
        R = np.abs(grad_E[i]) /  (k_ross * Energy)
        invR = 1 / R

        # Calc lambda, eq. 27
        coth = 1 / np.tanh(R)
        lamda = invR * (coth - invR)
        # Calc Flux, eq. 26
        Flux = - c_cgs * grad_E[i]  * lamda / k_ross
        
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

def doer_of_thing(fix, m, num = 2000):
    """
    Gives bolometric L 
    """
    tree_indexes, rays_T, rays_den, rays, radii, rays_vol = ray_maker(fix, m, num)
    _, _, rays_photo, rays_index_photo, tree_index_photo = get_photosphere(rays_T, rays_den, radii, tree_indexes)
    
    dim_ph = np.zeros(len(rays_index_photo))
    for j in range(len(rays_index_photo)):
        find_index_cell = int(rays_index_photo[j])
        vol_ph = rays_vol[j][find_index_cell]
        dim_ph[j] = (3 * vol_ph /(4 * np.pi))**(1/3) #in solar units
    dist_neigh = 2 * dim_ph
    # dist_neigh *= Rsol_to_cm #convert in CGS

    #Find the cell outside the photosphere and save its quantities
    # grad_E, energy_neigh_up, T_neigh_up, den_neigh_up  = find_neighbours(rays_T, rays_den, rays, radii, 
    #                                                                     rays_index_photo, dist_neigh)
    grad_E, energy_neigh_up, T_neigh_up, den_neigh_up  = find_neighbours(fix, m, tree_index_photo, dist_neigh)

    # Calculate Flux and see how it looks
    flux = flux_calculator(grad_E, energy_neigh_up, 
                           T_neigh_up, den_neigh_up)

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
    m = 6 # Choose BH
    fixes, days = select_fix(m)
    lums = []
            
    for idx in range(0,len(fixes)):
        lum = doer_of_thing(fixes[idx], m)
        lums.append(lum)
    
    if save:
        if alice:
            pre = '/home/s3745597/data1/TDE/'
            np.savetxt('red_backup_save'+ str(m) + '.txt', (days, lums))
            np.savetxt(pre + 'tde_comparison/data/alicered'+ str(m) + '.txt', (days, lums))
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

