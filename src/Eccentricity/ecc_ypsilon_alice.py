#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 21:06:49 2023

@author: konstantinos
"""

import numpy as np
import numba

# Custom Imports
from src.Extractors.time_extractor import days_since_distruption
from src.Calculators.casters import THE_SMALL_CASTER
from src.Eccentricity.eccentricity import  e_calc

pre1 = 'new'
check1 = 'tde_data2' + pre1 + '/'
fixes1 = np.arange(205, 219)

pre2 = 'hr4'
check2 = 'tde_data2' + pre2 + '/'
fixes2 = np.arange(204, 218)

@numba.njit
def masker(arr, mask):
    len_bound = np.sum(mask)
    new_arr = np.zeros(len_bound)
    k = 0
    for i in range(len(arr)):
        if mask[i]:
            new_arr[k] = arr[i]
            k += 1
    return new_arr
#%%
def maker(check, fix):
    # Constants
    G = 6.6743e-11 # SI
    Msol = 1.98847e30 # kg
    Rsol = 6.957e8 # m
    t = np.sqrt(Rsol**3 / (Msol*G )) # Follows from G=1
    m = 4
    Mbh = 10**m
    # Need these for the PW potential
    c = 3e8 * t/Rsol # c in simulator units.
    rg = 2*Mbh/c**2
    t_fall = 40 * (Mbh/1e6)**(0.5) # days EMR+20 p13
    Rt =  Mbh**(1/3) # Msol = 1, Rsol = 1
    apocenter = 2 * Rt * Mbh**(1/3)
    
    fix = str(fix)
    name = check + 'snap_' + fix + '/'
    
    day = str(np.round(days_since_distruption(name + '/snap_' + fix + '.h5'),3))
    # CM Position Data
    X = np.load(name +'CMx_' + fix + '.npy')
    Y = np.load(name +'CMy_' + fix + '.npy')
    Z = np.load(name +'CMz_' + fix + '.npy')
    Vx = np.load(name +'Vx_' + fix + '.npy')
    Vy = np.load(name +'Vy_' + fix + '.npy')
    Vz = np.load(name +'Vz_' + fix + '.npy')
    # Import Density
    Mass = np.load(name +'Mass_' + fix + '.npy')
     
    # Make Bound Mask
    R = np.sqrt( np.power(X,2) + np.power(Y,2)+ np.power(Z,2))
    V = np.sqrt( np.power(Vx,2) + np.power(Vy,2)+ np.power(Vz,2))
    Orbital = (0.5 * V**2 ) - Mbh / (R-rg)
    bound_mask = np.where(Orbital < 0, 1, 0)
    
    # Apply Mask
    X = masker(X, bound_mask)
    Y = masker(Y, bound_mask)
    Z = masker(Z, bound_mask)
    
    # Redefine to take account only the bound
    R = np.sqrt( np.power(X,2) + np.power(Y,2)+ np.power(Z,2))
    Vx = masker(Vx, bound_mask)
    Vy = masker(Vy, bound_mask)
    Vz = masker(Vz, bound_mask)
    Bound_Mass = masker(Mass, bound_mask)
    
    position = np.array((X,Y,Z)).T # Transpose for col. vectors
    velocity = np.array((Vx,Vy,Vz)).T 
    del X, Y, Z, Vx, Vy, Vz
    
    # EVOKE eccentricity
    _ , ecc = e_calc(position, velocity, Mbh)
    
    # Cast down to 100 values
    radii_start = np.log10(0.2*2*Rt)
    radii_stop = np.log10(apocenter) # apocenter


    radii = np.logspace(radii_start,  radii_stop, num = 200)
    mw_ecc_casted = THE_SMALL_CASTER(radii, R, ecc, weights = Bound_Mass,
                                   avg = True)
    
    one_minus_ecc = 1 - mw_ecc_casted
    
    return one_minus_ecc, radii, day, apocenter
        
#%%
def stacker(check, fixes):
    # First
    one_minus_ecc, radii, day_start, apocenter = maker(check, fixes[0])
    
    # Day holder
    days = [day_start]
       
    # Rest 
    for i in range(1, len(fixes)):
        one_minus_ecc_new, _, day, _ = maker(check, fixes[i])
        one_minus_ecc = np.add(one_minus_ecc, one_minus_ecc_new)
        days.append(day)

    # Mean
    inv_total_fixes = 1/len(fixes)
    one_minus_ecc = np.multiply(one_minus_ecc, inv_total_fixes)
    
    return one_minus_ecc, radii, days

# Do the thing
ecc_baseline, radii, days1 = stacker(check1, fixes1)
ecc_check, radii, days2 = stacker(check2, fixes2)
    
ypsilon = np.divide(ecc_baseline, ecc_check)
time = str(days1[0]) + '-' + str(days1[-1])

# Save to file
np.save('products/convergance/ecc_ypsilon-' + pre1 + '-' + pre2 + '-' + time, ypsilon)
np.save('products/convergance/ecc-' + pre1 + '-' + time, ecc_baseline)
np.save('products/convergance/ecc-' + pre2 + '-' + time, ecc_check)

