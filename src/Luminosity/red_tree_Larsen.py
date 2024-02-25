#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 2024

@author: paola 

Equations refer Steinberg&Stone22

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
import math

# Custom Imports
import src.Utilities.prelude as c
import src.Utilities.selectors as s
from src.Calculators.ray_forest import ray_maker_forest, ray_finder
from src.Luminosity.special_radii_tree import get_specialr
from astropy.coordinates import cartesian_to_spherical, spherical_to_cartesian

#%%
##
# FUNCTIONS
##
###


def outside_photo(snap, m, check, tree_index_photo, dist_neigh):
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
    Rad = np.load(pre + snap + '/Rad_' + snap + '.npy')
    Den = np.load(pre + snap + '/Den_' + snap + '.npy')
    
    # convert in CGS
    Rad *= Den 
    Rad *= c.en_den_converter
    Den *= c.den_converter

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
    # You have to shuft to move to the pericentre 
    x_low, y_low, z_low  = spherical_to_cartesian(r_low, theta_obs, phi_obs)
    x_low += Rt
    x_high, y_high, z_high  = spherical_to_cartesian(r_high, theta_obs, phi_obs)
    x_high += Rt 
    idx_low = np.zeros(len(tree_index_photo))
    idx_high = np.zeros(len(tree_index_photo))
    grad_r = np.zeros(len(tree_index_photo))
    magnitude = np.zeros(len(tree_index_photo))

    # make a tree with pericenter as origin
    # X -= Rt
    sim_value = [X, Y, Z] 
    sim_value = np.transpose(sim_value) # shape (number_points, 3). Need it for the tree.
    sim_tree = KDTree(sim_value)
   
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
        diff = diff / c.Rsol_to_cm # convert to CGS
        
        # Unitary vector in the r direction, for each observer
        rhat = [np.sin(theta_obs[i]) * np.cos(phi_obs[i]),
                np.sin(theta_obs[i]) * np.sin(phi_obs[i]),
                np.cos(theta_obs[i])
                ]
        grad_r[i] = np.dot(diff, rhat) # Project
        magnitude[i] = 1 / (np.linalg.norm(np.subtract(xyz_high, xyz_low)) * c.Rsol_to_cm) 
    
    # store data of neighbours
    idx_low = [int(x) for x in idx_low] #avoid dumb stuff with indexing later
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

    
def flux_calculator(gradr_E, magnitude, outside_energy, 
                    outside_temperature, outside_density, opacity_kind):
    """
    Get the flux for every observer.

    Parameters [CGS]
    ----------
    gradr_E: array.
            Radial component of the gradient of (radiation) energy density 
            across the photosphere (for every ray). 
    magnitude: array.
               Magnitude of the gradient of energy density. 
    outside_energy: array.
            Energy in the cell outside photosphere (for every ray). 
    outside_temperature: array.
            Temperature in the cell outside photosphere (for every ray). 
    outside_density: array.
            Density in the cell outside photosphere (for every ray). 
    opacity_kind: str.
            Choose the opacity.
        
    Returns [CGS]
    -------
    f: array
        Flux at every ray.
    """
    f = np.zeros(len(gradr_E))
    zero_count = 0
   
    # Ensure we can interpolate
    if opacity_kind == 'LTE':
        rho_low = np.exp(-45)
        Tmax = np.exp(17.87)  # 5.77e+07 K
        Tmin = np.exp(8.666)  # 5.8e+03 K
        from src.Opacity.LTE_opacity import opacity # NB: ln == False by default

    if opacity_kind == 'cloudy':
        Tmax = 1e13 
        Tmin = 316
        from src.Opacity.cloudy_opacity import old_opacity as opacity

    # compute flux at every ray
    for i in range(len(outside_energy)):
        Energy = outside_energy[i]
        Temperature = outside_temperature[i]
        Density = outside_density[i]
        mag = magnitude[i]

        # If here is nothing, light continues
        # if np.logical_and(m!=6, Density < rho_low):
        #     max_count += 1
        #     f[i] = max_travel
        #     if max_travel == 0:
        #         max_but_zero_count +=1
        #     continue
        
        # If stream, no light 
        if Temperature < Tmin:
            zero_count += 1
            f[i] = 0 
            continue
        
        # T too high => scattering
        if Temperature > Tmax:
            k_ross = opacity(Tmax, Density, 'scattering')
        else:    
            # Get Opacity, NOTE: Breaks Numba
            k_ross = opacity(Temperature, Density, 'rosseland') # CGS
        
        # Eq.(8) Steinberg&Stone22
        D = c.c / np.sqrt((3*k_ross)**2 + (mag/Energy)**2)
    
        Flux =  - D * gradr_E[i]
        
        f[i] = Flux

    print('Tot zeros:', zero_count)
    return f

def fld_luminosity(snap, m, check, thetas, phis, stops, num, opacity_kind):
    """
    Gives bolometric L 
    """
    # Make rays and find the photosphere
    rays = ray_maker_forest(snap, m, check, thetas, phis, 
                            stops, num, opacity_kind)   
    _, _, rays_photo, rays_index_photo, tree_index_photo = get_specialr(rays.T, rays.den, 
                                                            rays.radii, rays.tree_indexes, 
                                                            opacity_kind, select = 'photo') # CGS
    
    dim_ph = np.zeros(len(rays_index_photo))
    for j in range(len(rays_index_photo)):
        find_index_cell = int(rays_index_photo[j])
        vol_ph = rays.vol[j][find_index_cell]
        dim_ph[j] = (3 * vol_ph /(4 * np.pi))**(1/3) 
    dist_neigh = 2 * dim_ph # solar units
    # dist_neigh *= Rsol_to_cm #convert in CGS

    # Find the cell outside the photosphere and save its quantities
    grad_E, magnitude, energy_high, T_high, den_high  = outside_photo(snap, m, check,
                                                                    tree_index_photo, dist_neigh)

    # Calculate the flux 
    flux = flux_calculator(grad_E, magnitude, energy_high, 
                           T_high, den_high, opacity_kind)

    # Calculate luminosity 
    lum = np.zeros(len(flux))
    nan_count = 0
    neg_count = 0
    for i in range(len(flux)):
        lum_nodiff = 4 * np.pi * rays_photo[i]**2 * c.c * energy_high[i] 
        # Discard negative fluxes
        if flux[i] < 0:
            neg_count += 1
            lum_fld = 1e100
        elif math.isnan(flux[i]):
            nan_count += 1
            lum_fld = 1e100
        else:
            lum_fld = 4 * np.pi * rays_photo[i]**2 * flux[i] 

        lum[i] = min(lum_fld, lum_nodiff)
        
    # Average in observers
    lum = np.sum(lum)/192

    print('Negative:', neg_count)
    print('Nan:', nan_count)
    print(f'Snap {snap}, Lum %.3e' %lum, '\n---------' )
    return lum
#%%
##
# MAIN
##
if __name__ == "__main__":
    save = True
    m = 6 # Choose BH
    check = 'fid' # Choose fid // S60ComptonHires
    num = 1000
    opacity_kind = s.select_opacity(m)

    snapshots, days = s.select_snap(m, check)
    now = datetime.now()
    now = now.strftime("%d/%m/%Y %H:%M:%S")

    lums = np.zeros(len(snapshots))
    for idx in range(0,len(snapshots)):
        snap = snapshots[idx]
        print(f'Snapshot {snap}')
        filename = f"{m}/{snap}/snap_{snap}.h5"

        thetas, phis, stops, xyz_grid = ray_finder(filename)

        lum = fld_luminosity(snap, m, check, thetas, phis, stops, num, opacity_kind)
        lums[idx] = lum
    
    if save:
        if alice:
            pre_saving = f'/home/s3745597/data1/TDE/tde_comparison/data/alicered{m}{check}'
            with open(f'{pre_saving}Larsen_days.txt', 'a') as fdays:
                 fdays.write('# Run of ' + now + '\n#t/t_fb\n') 
                 fdays.write(' '.join(map(str, days)) + '\n')
                 fdays.close()
            with open(f'{pre_saving}Larsen.txt', 'a') as flum:
                 flum.write('# Run of ' + now + 't/t_fb\n')
                 flum.write(' '.join(map(str, lums)) + '\n')
                 flum.close()
        else:
             with open(f'data/red/Larsenreddata_m{m}{check}.txt', 'a') as flum:
                 flum.write('# Run of ' + now + '\n#t/t_fb\n') 
                 flum.write(' '.join(map(str, days)) + '\n')
                 flum.write('# Lum \n') 
                 flum.write(' '.join(map(str, lums)) + '\n')
                 flum.close() 



