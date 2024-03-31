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

from src.Utilities.isalice import isalice
alice, plot = isalice()

# Vanilla Imports
import numpy as np
import matplotlib.pyplot as plt
import h5py
import healpy as hp
from scipy.spatial import KDTree
from datetime import datetime

# Custom Imports
import src.Utilities.prelude as c
import src.Utilities.selectors as s
from src.Calculators.ray_forest import ray_maker_forest, ray_finder
from src.Luminosity.special_radii_tree import get_specialr
from astropy.coordinates import spherical_to_cartesian, cartesian_to_spherical

#%%
##
# FUNCTIONS
##
###

def find_neighbours(snap, m, check, tree_index_photo, dist_neigh):
    """
     For every ray, find the cells that are at +- fixed distance from photosphere.
     fixed distance = 2 * dimension of simulation cell at the photosphere

     Parameters
     ----------
     snap: int.
           Snapshot number.
     m: int.
        Exponent of BH mass
     check: str.
            Choose simualtion.
     tree_index_photo: 1D array.
                 Photosphere index in the tree.
     dist_neigh : 1D array.
                2(3πV/4)^(1/3), Distance from photosphere (in Rsol).

     Returns
     -------
     grad_Er: array.
             (Radiation) energy gradient for every ray at photosphere (CGS). 
     magnitude: array.
                Magnitude of grad_Er (CGS)
     energy_high: array.
             Energy for every ray in a cell outside photosphere (CGS). 
     T_high: array.
             Temperature for every ray in a cell outside photosphere (CGS). 
     den_high: array.
             Density for every ray in a cell outside photosphere (CGS). 

    """
    Mbh = 10**m 
    Rt =  Mbh**(1/3) # Msol = 1, Rsol = 1
    pre = s.select_prefix(m, check)
    snap = str(snap)
   
    # Import
    X = np.load(pre + snap + '/CMx_' + snap + '.npy') 
    Y = np.load(pre + snap + '/CMy_' + snap + '.npy')
    Z = np.load(pre + snap + '/CMz_' + snap + '.npy')
    T = np.load(pre + snap + '/T_' + snap + '.npy')
    Den = np.load(pre + snap + '/Den_' + snap + '.npy')
    Rad = np.load(pre +snap + '/Rad_' + snap + '.npy')
    
    # convert in CGS 
    Rad *= Den 
    Rad *= c.en_den_converter

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

    # convert to cartesian and query: BETTER TO USE OUR FIND_SPH_COORDINATE??
    x_low, y_low, z_low  = spherical_to_cartesian(r_low, theta_obs, phi_obs)
    x_high, y_high, z_high  = spherical_to_cartesian(r_high, theta_obs, phi_obs)
    # x_high += Rt # CHECK THIS!!!!!
    idx_low = np.zeros(len(tree_index_photo))
    idx_high = np.zeros(len(tree_index_photo))
    grad_r = np.zeros(len(tree_index_photo))
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
        diff /= c.Rsol_to_cm # convert to CGS
        
        # Unitary vector in the r direction, for each observer
        rhat = [np.sin(theta_obs[i]) * np.cos(phi_obs[i]), # xyz vextor
                np.sin(theta_obs[i]) * np.sin(phi_obs[i]),
                np.cos(theta_obs[i])
                ]
        grad_r[i] = np.dot(diff, rhat) # Project
        magnitude[i] = np.linalg.norm(np.divide(1, np.subtract(xyz_high, xyz_low) * c.Rsol_to_cm )) 

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
                    selected_temperature, selected_density, opacity_kind):
    """
    Get the flux for every observer.

    Parameters
    ----------
    grad_E: array.
            Energy gradient for every ray at photosphere. 
    magnitude: array.
                Magnitude of grad_Er (CGS)
    selected_energy: array.
            Energy for every ray in a cell outside photosphere. 
    selected_temperature: array.
            Temperature for every ray in a cell outside photosphere. 
    selected_density: array.
            Density for every ray in a cell outside photosphere. 
    opacity_kind: str.
            Choose the opacity.
        
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

    # Ensure we can interpolate
    rho_low = np.exp(-45)
    if opacity_kind == 'LTE':
        Tmax = np.exp(17.87)  # 5.77e+07 K
        Tmin = np.exp(8.666)  # 5.8e+03 K
        from src.Opacity.LTE_opacity import opacity # NB: ln == False by default

    if opacity_kind == 'cloudy':
        Tmax = 1e13 
        Tmin = 316
        from src.Opacity.cloudy_opacity import old_opacity as opacity

    for i in range(len(selected_energy)):
        Energy = selected_energy[i]
        max_travel = np.sign(-grad_E[i]) * c.c * Energy 
        
        Temperature = selected_temperature[i]
        Density = selected_density[i]

        # If here is nothing, light continues
        if np.logical_and(m!=6, Density < rho_low):
            max_count += 1
            f[i] = max_travel
            if max_travel == 0:
                max_but_zero_count +=1
            continue
        
        # If stream, no light 
        if Temperature < Tmin:
            zero_count += 1
            f[i] = 0 
            continue
        
        # T too high => scattering
        if Temperature > Tmax:
            Tscatter = np.exp(17.87)
            k_ross = opacity(Tscatter, Density, 'scattering')
        else:    
            # Get Opacity, NOTE: Breaks Numba
            # Paper says rosseland, page 22, below eq(3), but in the MATLAB 
            # script they actualy use red = absorption(rosseland) + scattering
            k_ross = opacity(Temperature, Density, 'rosseland')
        
        # Calc R, eq. 28
        R_kr = magnitude[i] /  (k_ross * Energy)
        print(magnitude[i]/Energy)
        invR = 1 / R_kr
        R_kr = float(R_kr) # to avoid dumb thing with tanh(R)
    
        # Calc lambda, eq. 27
        coth = 1 / np.tanh(R_kr)
        lamda = invR * (coth - invR)
        # Calc Flux, eq. 26
        Flux =  - c.c * grad_E[i]  * lamda / k_ross
        
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

def doer_of_thing(snap, m, check, thetas, phis, stops, num, opacity_kind):
    """
    Gives bolometric L 
    """
    rays = ray_maker_forest(snap, m, check, thetas, phis, stops, 
                            num, opacity_kind, star = False)    
    _, _, rays_photo, rays_index_photo, tree_index_photo = get_specialr(rays.T, rays.den, rays.radii, 
                                                            rays.tree_indexes, opacity_kind, select = 'photo')
    # Calculate the distance of the neighbours considered
    size_of_ph = np.zeros(len(rays_index_photo))
    for j in range(len(rays_index_photo)):
        find_index_cell = int(rays_index_photo[j])
        vol_ph = rays.vol[j][find_index_cell]
        size_of_ph[j] = (3 * vol_ph /(4 * np.pi))**(1/3) #in solar units
    dist_neigh = 2 * size_of_ph
    # dist_neigh *= Rsol_to_cm #convert in CGS

    # Find the cell outside the photosphere and save its quantities
    grad_E, magnitude, energy_high, T_high, den_high  = find_neighbours(snap, m, check,
                                                                        tree_index_photo, dist_neigh)

    # Calculate Flux and see how it looks
    flux = flux_calculator(grad_E, magnitude, energy_high, 
                           T_high, den_high, opacity_kind)

    # Calculate luminosity 
    lum = np.zeros(len(flux))
    zero_count = 0
    neg_count = 0
    for i in range(len(flux)):
        # Turn to luminosity
        if np.abs(flux[i]) < 1e-3:
            zero_count += 1
        if flux[i] < 0:
            neg_count += 1
            flux[i] = 0 

        lum[i] = flux[i] * 4 * np.pi * rays_photo[i]**2
        # add an alice backup here
    # Average in observers
    lum = np.sum(lum)/192

    print('Tot zeros:', zero_count)
    print('Negative: ', neg_count)      
    print('Snap %i' %snap, ', Lum %.3e' %lum, '\n---------' )
    return lum
#%%
##
# MAIN
##
if __name__ == "__main__":
    save = True
    m = 5 # Choose BH
    mstar = 0.5
    rstar = 0.47
    check = 'fid' # Choose fid // S60ComptonHires
    num = 1000
    opacity_kind = s.select_opacity(m)

    snapshots, days = s.select_snap(m, mstar, rstar, check)
    now = datetime.now()
    now = now.strftime("%d/%m/%Y %H:%M:%S")

    lums = np.zeros(len(snapshots))
    for idx in range(0,len(snapshots)):
        snap = snapshots[idx]
        print(f'Snapshot {snap}')
        if alice:
            pre = '/home/s3745597/data1/TDE/'
            filename = pre + f'{m}-{check}/snap_{snap}/snap_{snap}.h5'
        else:
            filename = f"{m}/{snap}/snap_{snap}.h5"

        thetas, phis, stops, xyz_grid = ray_finder(filename)

        lum = doer_of_thing(snap, m, check, thetas, phis, stops, num, opacity_kind)
        lums[idx] = lum
        print(lum)
    
    if save:
        if alice:
            pre_saving = f'/home/s3745597/data1/TDE/tde_comparison/data/alicered{m}{check}'
            with open(f'{pre_saving}_days.txt', 'a') as fdays:
                 fdays.write('# Run of ' + now + '\n#t/t_fb\n') 
                 fdays.write(' '.join(map(str, days)) + '\n')
                 fdays.close()
            with open(f'{pre_saving}.txt', 'a') as flum:
                 flum.write('# Run of ' + now + 't/t_fb\n')
                 flum.write(' '.join(map(str, lums)) + '\n')
                 flum.close()
        else:
             with open(f'data/red/reddata_m{m}{check}.txt', 'a') as flum:
                 flum.write('# Run of ' + now + '\n#t/t_fb\n') 
                 flum.write(' '.join(map(str, days)) + '\n')
                 flum.write('# Lum \n') 
                 flum.write(' '.join(map(str, lums)) + '\n')
                 flum.close() 

    # %% Plotting
    # if plot:
    #     plt.figure()
    #     plt.plot(lums, '-o', color = 'maroon')
    #     plt.yscale('log')
    #     plt.ylabel('Bolometric Luminosity [erg/s]')
    #     plt.xlabel('Days')
    #     if m == 6:
    #         plt.title('FLD for $10^6 \quad M_\odot$')
    #         plt.ylim(1e41,1e45)
    #     if m == 4:
    #         plt.title('FLD for $10^4 \quad M_\odot$')
    #         plt.ylim(1e39,1e42)
    #     plt.grid()
    #     #plt.savefig(f'Final plot/ourred{m}.png')
    #     plt.show()

