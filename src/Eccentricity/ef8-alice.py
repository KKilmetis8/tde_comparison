#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 18:46:16 2023

@author: konstantinos
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [8.0, 4.0]
import numba
from src.Eccentricity.eccentricity import  e_calc
from src.Extractors.time_extractor import days_since_distruption
from src.Calculators.casters import THE_SMALL_CASTER

# Choose sims
bh = '6' # or '4

# Snapshot specification
final4 = 412
final6 = 1008
frames = 200
start4 = final4 - frames
start6 = final6 - frames

if bh == '6':
    Mbh = 1e6 # * Msol
    Rt =  Mbh**(1/3) # Msol = 1, Rsol = 1
    t_fall = 40 * (Mbh/1e6)**(0.5) # days EMR+20 p13
    
    pre = 'tde_data/snap_'
    fixes = np.arange(start6,final6+1) # 750,1008+1 , 225,412+1
    radii_start = np.log10(0.2*2*Rt)
    radii_stop = np.log10(100*2*Rt) # apocenter
if bh == '4':
    Mbh = 1e4 # * Msol
    Rt =  Mbh**(1/3) # Msol = 1, Rsol = 1
    t_fall = 40 * (Mbh/1e6)**(0.5) # days EMR+20 p13
    pre = 'tde_data2/snap_'
    fixes = np.arange(start4,final4+1) # 750,1008+1 , 225,412+1
    radii_start = np.log10(0.2*2*Rt)
    radii_stop = np.log10(22*2*Rt) # apocenter

# Define radii to project to
pixel_num = 100

# Constants
G = 6.6743e-11 # SI
Msol = 1.98847e30 # kg
Rsol = 6.957e8 # m
t = np.sqrt(Rsol**3 / (Msol*G )) # Follows from G=1
# Need these for the PW potential
c = 3e8 * t/Rsol # c in simulator units.
rg = 2*Mbh/c**2

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
colarr = []
fixdays = []
for fix in fixes:
     fix = str(fix)
     X = np.load(pre + fix + '/CMx_' + fix + '.npy')
     Y = np.load(pre + fix + '/CMy_' + fix + '.npy')
     Z = np.load(pre + fix + '/CMz_' + fix + '.npy')
     Vx = np.load(pre + fix + '/Vx_' + fix + '.npy')
     Vy = np.load(pre + fix + '/Vy_' + fix + '.npy')
     Vz = np.load(pre + fix + '/Vz_' + fix + '.npy')
     Mass = np.load(pre + fix + '/Mass__' + fix + '.npy')
     
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

     radii = np.logspace(radii_start,  radii_stop, num = pixel_num)
     mw_ecc_casted = THE_SMALL_CASTER(radii, R, ecc, 
                                      weights = Bound_Mass, avg = True)
     colarr.append(mw_ecc_casted)
     
     # Day
     day = np.round(days_since_distruption(pre + fix +'/snap_' + fix + '.h5'),1)
     fixdays.append(day)

eccs = [fixdays, colarr]
np.save('ecc'+bh, eccs)
