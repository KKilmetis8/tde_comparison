#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 15:47:48 2023

@author: konstantinos, Paola
"""
import numpy as np
import numba

# Just to understand what it does
# we never call this actually
def Haversine(observers, cell_theta, cell_phi):
    ''' Distance on a spherical surface.
    
        NOTE: For us latitude is θ and longitude is φ
              instead of φ and λ
    '''
    
    c = 8_415 # Arbitrary
    
    for i, observer in enumerate(observers):
        obs_theta = observer[0]
        obs_phi = observer[1]
        delta_theta = obs_theta - cell_theta
        delta_phi = obs_phi - cell_phi
        
        # Haversine formula
        a = np.sin(delta_theta / 2)**2 + \
            np.cos(obs_theta) * np.cos(cell_theta) * \
            np.sin(delta_phi/2)**2
        new_c = 2 * np.arctan2( np.sqrt(a), np.sqrt(1-a))
        # print(new_c)
        # For us r = 1
        if new_c < c:
            c = new_c
            closest_observer = i # Index of closest obs
    return closest_observer

@numba.njit
def THE_SPHERICAL_CASTER(radii, R,
               observers, THETA, PHI, 
               Den,
               weights=None, avg=False, loud = False):
    
    gridded_density = np.zeros((len(radii), len(observers)))
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
        #if type(weights) != type(None):
        gridded_density[idx_r, idx_obs] += Den[i] * weights[i]
        gridded_weights[idx_r, idx_obs] += weights[i]
        # else:
        #     gridded_density[idx_r, idx_obs] += Den[i]
            
        # Progress check
        progress = int(np.round(i/len(R),1) * 100)
        if i % 100  == 0 and progress != current_progress:
            if loud:
                print('THE CASTER IS', progress, '% DONE')
            current_progress = progress
                
    # Normalize
    final_density = gridded_density
    if avg:
        final_density = np.divide(gridded_density,counter)
    # if type(weights) != type(None):
    final_density = np.divide(gridded_density, gridded_weights)
    return final_density