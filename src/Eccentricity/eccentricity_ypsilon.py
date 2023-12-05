#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 20:40:13 2023

@author: konstantinos
"""

import numpy as np
import numba

# Custom Imports
from src.Extractors.time_extractor import days_since_distruption
from src.Calculators.casters import THE_SMALL_CASTER
from src.Eccentricity.eccentricity import  e_calc

# Pretty plots
import matplotlib.pyplot as plt
import colorcet # cooler colormaps
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [6 , 8]
AEK = '#F1C410' # Important color


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
def maker(check, sim, fix):
    # Constants
    G = 6.6743e-11 # SI
    Msol = 1.98847e30 # kg
    Rsol = 6.957e8 # m
    t = np.sqrt(Rsol**3 / (Msol*G )) # Follows from G=1
    Mbh = 1e4 # * Msol
    # Need these for the PW potential
    c = 3e8 * t/Rsol # c in simulator units.
    rg = 2*Mbh/c**2
    t_fall = 40 * (Mbh/1e6)**(0.5) # days EMR+20 p13
    
    fix = str(fix)
    name = 'convergence/' + fix + '/' + sim + fix
        
    X = np.load(name +'CMx.npy')
    Y = np.load(name + 'CMy.npy')
    Z = np.load(name + 'CMz.npy')
    Vx = np.load(name +'Vx.npy')
    Vy = np.load(name + 'Vy.npy')
    Vz = np.load(name + 'Vz.npy')
    # Import Density
    Den = np.load(name + 'Den.npy')
    Mass = np.load(name + 'Mass.npy')
     
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
    Rt =  Mbh**(1/3) # Msol = 1, Rsol = 1
    t_fall = 40 * (Mbh/1e6)**(0.5) # days EMR+20 p13
    radii_start = np.log10(0.2*2*Rt)
    radii_stop = np.log10(22*2*Rt) # apocenter
    apocenter = 22*2*Rt

    radii = np.logspace(radii_start,  radii_stop, num = 100)
    mw_ecc_casted = THE_SMALL_CASTER(radii, R, ecc, weights = Mass,
                                   avg = True)
    
    # mw_ecc_casted = np.nan_to_num(mw_ecc_casted)
    one_minus_ecc = 1 - mw_ecc_casted
    
    day = np.round(days_since_distruption(name + '.h5'),1)
    # t_by_tfb = day/t_fall
    # fixdays.append(t_by_tfb)
    
    return one_minus_ecc, radii, day, apocenter
        
#%%
check1 = 'base'
check2 = 'hr2'
# check3 = 'hr2'
sims = [check1 + '-', check2 + '-' ] # check3 + '-']
fixes = [445]

for fix in fixes:
    ecc_baseline, radii, day1, apocenter = maker(check1, sims[0], fix)
    ecc_check, _, day2, _ = maker(check2, sims[1], fix)
#    ecc_check3, _, day3, _ = maker(check2, sims[2], fix)
    ypsilon = np.divide(ecc_baseline, ecc_check)
#%% Plotting
fig, ax = plt.subplots(2,1, tight_layout = True, sharex = True)

# Images
ax[0].plot(radii/apocenter, ypsilon, 
         '-h', color = 'maroon', label = '$10^4 M_\odot$', 
         markersize = 5)
ax[1].plot(radii/apocenter, ecc_baseline, 
         '-h', color = AEK, label = check1,
         markersize = 5)
ax[1].plot(radii/apocenter, ecc_check, 
         '-h', color = 'k', label = check2,
         markersize = 5)
# ax[1].plot(radii/apocenter, ecc_check3, 
#          '-h', color = 'maroon', label = check3,
#          markersize = 5)

ax[0].grid()
ax[1].grid()
ax[1].legend()
#
fig.suptitle('Eccentricity Check', fontsize = 25)
ax[0].set_ylabel(r'$\upsilon$ 1-$e_{base}$ / 1-$e_{s30}$')
ax[1].set_ylabel('1-e')
# ax[1].text(0.1, 0.1, 'Baseline: ' + day1,
#             fontsize = 50,
#             color='white', fontweight = 'bold', 
#             transform=ax[1].transAxes)
# ax[2].text(0.1, 0.1, 'S30: ' + day2,
#             fontsize = 50,
#             color='white', fontweight = 'bold', 
#             transform=ax[2].transAxes)
plt.xlabel(r'Radius [$r / R_a$]', fontsize = 14)
plt.xscale('log')
