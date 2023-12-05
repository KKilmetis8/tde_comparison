#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 22:46:20 2023

@author: konstantinos
"""
import numpy as np

# Pretty plots
import matplotlib.pyplot as plt
import colorcet # cooler colormaps
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [6 , 4]
AEK = '#F1C410' # Important color

# Data load
check1 = 'new'
check2 = 's10'

# HR4
# days = '2.373-2.879'
# days = '3.027-3.506' 
# days = '3.58-3.912'

# Compton
# days = '1.746-2.41'
# days = '2.447-3.137'

# S10
days = '1.494-1.968'
#days = '2.041-2.632'

ypsilon = np.load('products/convergance/ecc_ypsilon-' + check1 + '-' + check2 + '-' + days + '.npy')
ecc_baseline = np.load('products/convergance/ecc-' + check1 + '-' + days + '.npy')
ecc_check = np.load('products/convergance/ecc-' + check2 + '-' + days + '.npy')

Mbh = 1e4 # * Msol
Rt =  Mbh**(1/3) # Msol = 1, Rsol = 1
apocenter = 2 * Rt * Mbh**(1/3)
radii_start = np.log10(0.2*2*Rt)
radii_stop = np.log10(apocenter) # apocenter
radii = np.logspace(radii_start,  radii_stop, num = 200)

#%% Plotting
fig, ax = plt.subplots(1,2, tight_layout = True, sharex = True)

# Images
ax[0].plot(radii/apocenter, ypsilon, 
         '-', color = 'maroon', label = '$10^4 M_\odot$', 
         markersize = 5)
ax[1].plot(radii/apocenter, ecc_baseline, 
         '-', color = AEK, label = 'Fid',
         markersize = 5)
ax[1].plot(radii/apocenter, ecc_check, 
         '--', color = 'k', label = check2,
         markersize = 5)
ax[0].grid()
ax[1].grid()
ax[1].legend()
#
fig.suptitle('Eccentricity Fid - S10 | Days: ' + days, fontsize = 25)
ax[0].set_ylabel(r'$\upsilon$ 1-$e_{base}$ / 1-$e_{s30}$')
ax[1].set_ylabel('1-e')
plt.xlabel(r'Radius [$r / R_a$]', fontsize = 14)
plt.xscale('log')