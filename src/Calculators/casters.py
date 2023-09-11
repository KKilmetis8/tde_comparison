#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 15:27:54 2023

@author: konstantinos
"""
import numpy as np
import numba

@numba.njit
def THE_CASTER(radii, R,
               thetas, THETA,
               Den, 
               weights=None, avg=False):
    '''
    Casts the density down to a smaller size vector

    Parameters
    ----------
    radii : arr,
        Array of radii we want to cast to.
    R : arr,
        Radius data.
    thetas: arr,
        Array of true anomalies we want to cast to.
    THETA: arr,
        True anomalies data
    Den: arr,
        Density data
        
    Returns
    -------
     density: arr
        Cast down version of density

    '''
    gridded_density = np.zeros((len(radii), len(thetas)))
    gridded_weights = np.zeros((len(radii), len(thetas)))
    counter = np.zeros((len(radii), len(thetas)))
    current_progress = 0
    for i in range(len(R)):
        
        # Check how close the true R is to our radii
        diffs = np.abs(radii - R[i])
        # Get the index of the minimum
        idx_r = np.argmin(diffs)
        
        # Same for true anomaly
        diffs = np.abs(thetas - THETA[i])
        idx_t = np.argmin(diffs)
        
        # Add to grid
        counter[idx_r, idx_t] += 1
        
        # Add to grid | weighted mean
        if weights != None:
            gridded_density[idx_r, idx_t] += Den[i] * weights[i]
            gridded_weights[idx_r, idx_t] += weights[i]
        else:
            gridded_density[idx_r, idx_t] += Den[i]
        
        # Progress check
        progress = int(np.round(i/len(R),1) * 100)
        if i % 100  == 0 and progress != current_progress:
            print('THE CASTER IS', progress, '% DONE')
            current_progress = progress
            
    # Normalize
    final_density = gridded_density
    if avg:
        final_density = np.divide(gridded_density,counter)
    if weights != None:
        final_density = np.divide(gridded_density, gridded_weights)
    return final_density

@numba.njit
def THE_SMALL_CASTER(radii, R,
               Den, weights=None, avg = False, loud=False):
    '''
    Casts the density down to a smaller size vector

    Parameters
    ----------
    radii : arr,
        Array of radii we want to cast to.
    R : arr,
        Radius data.
    Den: arr,
        Density data
        
    Returns
    -------
     density: arr
        Cast down version of density

    '''
    gridded_density = np.zeros((len(radii)))
    gridded_weights = np.zeros((len(radii)))
    counter = np.zeros((len(radii)))
    current_progress = 0
    for i in range(len(R)):
        
        # Check how close the true R is to our radii
        diffs = np.abs(radii - R[i])
        # Get the index of the minimum
        idx_r = np.argmin(diffs)
        
        # Add to grid | weighted mean
        if weights != None:
            gridded_density[idx_r] += Den[i] * weights[i]
            gridded_weights[idx_r] += weights[i]
        else:
            gridded_density[idx_r] += Den[i]
        
        counter[idx_r] += 1
        # Progress check
        if loud:
            progress = int(np.round(i/len(R),1) * 100)
            if i % 100  == 0 and progress != current_progress:
                print('THE small CASTER IS', progress, '% DONE')
                current_progress = progress
    
    # Normalize
    final_density = gridded_density
    if avg:
        final_density = np.divide(gridded_density, counter)
    if weights != None:
        final_density = np.divide(gridded_density, gridded_weights)
    return final_density

@numba.njit
def THE_TRIPLE_CASTER(xs, X,
               ys, Y,
               zs, Z,
               Den, 
               weights = None, avg = False):
    
    gridded_density = np.zeros(( len(xs), len(ys), len(zs) ))
    gridded_weights = np.zeros(( len(xs), len(ys), len(zs) ))
    counter = np.zeros(( len(xs), len(ys), len(zs) ))
    current_progress = 0
    for i in range(len(X)):
        
        # Check how close the true R is to our radii
        diffs = np.abs(xs - X[i])
        # Get the index of the minimum
        idx_x = np.argmin(diffs)
        # Same for thetas
        diffs = np.abs(ys - Y[i])
        idx_y = np.argmin(diffs)
        # Same for phis
        diffs = np.abs(zs - Z[i])
        idx_z = np.argmin(diffs)
        
        # Add to grid
        if weights != None:
            gridded_density[idx_x, idx_y, idx_z] += Den[i] * weights[i]
            gridded_weights[idx_x, idx_y, idx_z] += weights[i]
        else:
            gridded_density[idx_x, idx_y, idx_z] += Den[i]
        
        counter[idx_x, idx_y, idx_z] += 1
        
        # Progress check
        progress = int(np.round(i/len(X),1) * 100)
        if i % 100  == 0 and progress != current_progress:
            print('THE CASTER IS', progress, '% DONE')
            current_progress = progress
            
    final_density = gridded_density
    if avg:
        final_density = np.divide(gridded_density,counter)
    if weights != None:
        final_density = np.divide(gridded_density, gridded_weights)
    return final_density
