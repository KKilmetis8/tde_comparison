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
import numba
from scipy.spatial import KDTree
from datetime import datetime
# Custom Imports
from src.Opacity.opacity_table import opacity
from src.Calculators.ray_tree import ray_maker
from src.Luminosity.special_radii_tree import get_photosphere
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
#Rsol_to_cm = 6.957e10 # [cm]
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
            snapshots = [844, 881, 925, 950, 1008] 
            days = [1, 1.1, 1.3, 1.4, 1.608] 
    return snapshots, days

def find_neighbours(fix, m, rays_index_photo):
    X = np.load( str(m) + '/'  + str(fix) + '/CMx_' + str(fix) + '.npy')
    Y = np.load( str(m) + '/'  + str(fix) + '/CMy_' + str(fix) + '.npy')
    Z = np.load( str(m) + '/'  + str(fix) + '/CMz_' + str(fix) + '.npy')
    Rad = np.load(str(m) + '/'  +str(fix) + '/Rad_' + str(fix) + '.npy')
    T = np.load( str(m) + '/'  + str(fix) + '/T_' + str(fix) + '.npy')
    Den = np.load( str(m) + '/'  + str(fix) + '/Den_' + str(fix) + '.npy')
    Rad *= Den 
    Rad *= en_den_converter

    # make a tree
    sim_value = [X, Y, Z] 
    sim_value = np.transpose(sim_value) #array of dim (number_points, 3)
    sim_tree = KDTree(sim_value)

    rays_index_photo = [int(x) for x in rays_index_photo]
    xyz_selected = [X[rays_index_photo], Y[rays_index_photo], Z[rays_index_photo]]
    r_selected = np.sqrt(xyz_selected[0]**2 + xyz_selected[1]**2 + xyz_selected[2]**2)
    xyz_selected = np.transpose(xyz_selected) #you need it to query
    energy_selected = Rad[rays_index_photo]

    xyz_neigh_high = []
    dist_neigh_high = np.zeros(len(r_selected))
    idx_neigh_high = np.zeros(len(r_selected))
    r_neigh_high = np.zeros(len(r_selected))
    energy_neigh_high = np.zeros(len(r_selected))
    T_neigh_high = np.zeros(len(r_selected))
    den_neigh_high = np.zeros(len(r_selected))

    xyz_neigh_low = []
    dist_neigh_low = np.zeros(len(r_selected))
    idx_neigh_low = np.zeros(len(r_selected))
    r_neigh_low = np.zeros(len(r_selected)) 
    energy_neigh_low = np.zeros(len(r_selected))
    T_neigh_low = np.zeros(len(r_selected))
    den_neigh_low = np.zeros(len(r_selected))

    for j in range(len(r_selected)):
        i = 1
        count_high = 0
        count_low = 0
        if r_selected[j] < 5:
            print('low photo', j)
            count_low = 2
            r_neigh_low[j] = r_selected[j]
            idx_neigh_low[j] = rays_index_photo[j]
            xyz_neigh_low.append(xyz_selected[j])
            energy_neigh_low[j] = energy_selected[j]
            T_neigh_low[j] = T[rays_index_photo[j]]
            den_neigh_low[j] = Den[rays_index_photo[j]]
        while (count_high !=2 or count_low !=2):
            dist_test, idx_test = sim_tree.query(xyz_selected[j], k = [i]) # find the 2nd nearest neighbours 
            xyz_test = [X[idx_test], Y[idx_test], Z[idx_test]]
            r_test = np.sqrt(xyz_test[0]**2 + xyz_test[1]**2 + xyz_test[2]**2)
            #print(r_test)
            if np.logical_and(r_test > r_selected[j], count_high < 2):
                count_high += 1
                if count_high == 2:
                    r_neigh_high[j] = r_test
                    dist_neigh_high[j] = dist_test
                    idx_neigh_high[j] = idx_test
                    xyz_neigh_high.append(xyz_test)
                    energy_neigh_high[j] = Rad[idx_test]
                    T_neigh_high[j] = T[idx_test]
                    den_neigh_high[j] = Den[idx_test]
            elif np.logical_and(r_test < r_selected[j], count_low < 2):
                count_low += 1
                if count_low == 2:
                    r_neigh_low[j] = r_test
                    dist_neigh_low[j] = dist_test
                    idx_neigh_low[j] = idx_test
                    xyz_neigh_low.append(xyz_test)
                    energy_neigh_low[j] = Rad[idx_test]
                    T_neigh_low[j] = T[idx_test]
                    den_neigh_low[j] = Den[idx_test]
            i += 1

    deltadist = r_neigh_low + r_neigh_high #den_neigh_low + den_neigh_high
    deltaE = energy_neigh_high - energy_neigh_low
    grad_E = deltaE / deltadist
    # r_leaf = np.sqrt(x_selected[i]**2 + y_selected[i]**2 + z_selected[i]**2)
    # _, idx = sim_tree.query(leaf, k = 4)
    # x_neigh = X[idx]
    # y_neigh = Y[idx]
    # z_neigh = Z[idx]
    # r_neigh = np.sqrt(x_neigh[i]**2 + y_neigh[i]**2 + z_neigh[i]**2)
    # idx_lower = np.argmin(r_neigh-r_leaf) #we want it to be negative so the neighbour is before photo
    # idx_lower = np.argmax(r_neigh-r_leaf) #we want it to be positive so the neighbour is after photo

    return grad_E, energy_neigh_high, T_neigh_high, den_neigh_high

    
def flux_calculator(grad_E, selected_energy, 
                    selected_temperature, selected_density):
    """
    Get the flux for every observer.
    Everything is in CGS.

    Parameters: 
    grad_E, idx_tot are 1D-array of lenght = len(rays)
    rays, rays_T, rays_den are len(rays) x N_cells arrays
    """
    f = np.zeros(len(grad_E))
    max_count = 0
    max_but_zero_count = 0
    zero_count = 0
    flux_zero = 0
    flux_count = 0

    for i in range(len(selected_energy)):
        # We compute stuff OUTSIDE the photosphere
        # (which is at index idx_tot[i])
        Energy = selected_energy[i]
        max_travel = np.sign(-grad_E[i]) * c_cgs * Energy # or should we keep the abs???
        
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

def doer_of_thing(fix, m):
    """
    Gives bolometric L and R_ph (of evry observer)
    """
    tree_indexes, rays_T, rays_den, _, radii, _ = ray_maker(fix, m)
    _, _, rays_photo, rays_index_photo = get_photosphere(rays_T, rays_den, radii, tree_indexes)

    #Find the neighbour to photosphere and save its quantities
    grad_E, energy_neigh_up, T_neigh_up, den_neigh_up  = find_neighbours(fix, m, rays_index_photo)
    
    # Calculate gradE for all rays
    #deltaE = energy_neigh - energy_selected
    #grad_E = (deltaE) / dist_neigh

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
    return lum, rays_photo
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
        lum, sphere_radius = doer_of_thing(fixes[idx], m)
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
        #plt.savefig('Final plot/new_red' + str(m) + '.png')
        plt.show()

