#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 10:23:43 2023

@author: konstantinos
"""
import numpy as np
import numba

@numba.njit
def THROUPLE_S_CASTERS(radii, R,
               observers, THETA, PHI, 
               T, Den, Rad,
               weights=None, avg=True, loud = False):
    
    gridded_density = np.zeros((len(radii), len(observers)))
    gridded_temperature = np.zeros((len(radii), len(observers)))  
    gridded_rad = np.zeros((len(radii), len(observers)))
    gridded_weights = np.zeros((len(radii), len(observers)))
    counter = np.zeros((len(radii), len(observers)))
    current_progress = 0
    
    for i in range(len(R)): # Loop over cell      
        # Check how close the true R is to our radii
        diffs = np.abs(radii - R[i])
        # Get the index of the minimum
        idx_r = np.argmin(diffs)
        
        # Use Haversine formula to calculate distance on a sphere
        cell_theta = THETA[i]
        cell_phi = PHI[i]

        c = 8_415 # Arbitrary big
        for j, observer in enumerate(observers):
            obs_theta = observer[0]
            obs_phi = observer[1]
            delta_theta = obs_theta - cell_theta
            delta_phi = obs_phi - cell_phi
            
            # Haversine formula (r = 1)
            a = np.sin(delta_theta / 2)**2 + np.cos(obs_theta) * np.cos(cell_theta) * np.sin(delta_phi/2)**2
            new_c = 2 * np.arctan2( np.sqrt(a), np.sqrt(1-a))

            if new_c < c:
                c = new_c
                idx_obs = j # Index of closest obs
                
        # For averaging
        counter[idx_r, idx_obs] += 1
        
        # Add to grid | weighted mean
        gridded_density[idx_r, idx_obs] += Den[i] * weights[i]
        gridded_temperature[idx_r, idx_obs] += T[i] * weights[i]       
        gridded_rad[idx_r, idx_obs] += Rad[i] * weights[i]
        gridded_weights[idx_r, idx_obs] += weights[i]

        # Progress check
        progress = int(np.round(i/len(R),1) * 100)
        if i % 100  == 0 and progress != current_progress:
            if loud:
                print('THE CASTER IS', progress, '% DONE')
            current_progress = progress
                
    # Normalize
    final_density = gridded_density
    final_temperature = gridded_temperature
    final_rad = gridded_rad
    if avg:
        final_density = np.divide(gridded_density, counter)
        final_temperature = np.divide(gridded_temperature, counter)
        final_rad = np.divide(gridded_rad, counter)

    # Divide by weights
    final_density = np.divide(final_density, gridded_weights)
    final_temperature = np.divide(final_temperature, gridded_weights)    
    final_rad = np.divide(final_rad, gridded_weights)

    return final_temperature, final_density, final_rad

@numba.njit
def COUPLE_S_CASTERS(radii, R,
               observers, THETA, PHI, 
               T, Den, 
               weights=None, avg=False, loud = False):
    
    gridded_density = np.zeros((len(radii), len(observers)))
    gridded_temperature = np.zeros((len(radii), len(observers)))  
    gridded_weights = np.zeros((len(radii), len(observers)))
    counter = np.zeros((len(radii), len(observers)))
    current_progress = 0
    for i in range(len(R)): # Loop over cell      
        # Check how close the true R is to our radii
        diffs = np.abs(radii - R[i])
        # Get the index of the minimum
        idx_r = np.argmin(diffs)
        
        # Use Haversine formula to calculate distance on a sphere
        cell_phi = PHI[i]
        cell_theta = THETA[i]

        c = 8_415 # Arbitrary
        for j, observer in enumerate(observers):
            obs_theta = observer[0]
            obs_phi = observer[1]
            delta_theta = obs_theta - cell_theta
            delta_phi = obs_phi - cell_phi
            
            # Haversine formula
            a = np.sin(delta_theta / 2)**2 + np.cos(obs_theta) * np.cos(cell_theta) * np.sin(delta_phi/2)**2
            new_c = 2 * np.arctan2( np.sqrt(a), np.sqrt(1-a))
            # print(new_c)
            # For us r = 1
            if new_c < c:
                c = new_c
                idx_obs = j # Index of closest obs
                
        # For averaging
        counter[idx_r, idx_obs] += 1
        
        # Add to grid | weighted mean
        gridded_density[idx_r, idx_obs] += Den[i] * weights[i]
        gridded_temperature[idx_r, idx_obs] += T[i]  * weights[i]       
        gridded_weights[idx_r, idx_obs] += weights[i]
        
        # Progress check
        progress = int(np.round(i/len(R),1) * 100)
        if i % 100  == 0 and progress != current_progress:
            if loud:
                print('THE CASTER IS', progress, '% DONE')
            current_progress = progress
                
    # Normalize
    final_density = gridded_density
    final_temperature = gridded_temperature

    if avg:
        final_density = np.divide(gridded_density,counter)
        final_temperature = np.divide(gridded_temperature,counter)

    # Divide by weights
    final_density = np.divide(final_density, gridded_weights)
    final_temperature = np.divide(final_temperature, gridded_weights)    

    return final_temperature, final_density

