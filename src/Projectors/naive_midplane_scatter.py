#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 13:21:47 2024

@author: konstantinos
"""

import sys
sys.path.append('/Users/paolamartire/tde_comparison')

from src.Utilities.isalice import isalice
alice, plot = isalice()

import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from src.Utilities.selectors import select_prefix
import colorcet
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.figsize'] = [6, 4]
plt.rcParams['axes.facecolor']= 	'whitesmoke'
#%% Constants & Converter
Msol_to_g = 1.989e33 # [g]
Rsol_to_cm = 6.957e10 # [cm]
den_converter = Msol_to_g / Rsol_to_cm**3

# def grid_maker(fix, m, check, what, mass_weigh, x_num, y_num, z_num = 100):
fix = 844
m = 6
what = 'temperature'
fix = str(fix)

Mbh = 10**m 
Rt =  Mbh**(1/3) # Msol = 1, Rsol = 1
apocenter = 2 * Rt * Mbh**(1/3)  # There is m_* hereeee

# Load data
# pre = select_prefix(m, check)
pre = str(m) + '/'
X = np.load(pre + fix + '/CMx_' + fix + '.npy')
Y = np.load(pre + fix + '/CMy_' + fix + '.npy')
Z = np.load(pre + fix + '/CMz_' + fix + '.npy')
# if mass_weigh:
#     Mass = np.load(pre + fix + '/Mass_' + fix + '.npy')    

if what == 'density':
    projected_quantity = np.load(pre + fix + '/Den_' + fix + '.npy')
    projected_quantity *= den_converter 
elif what == 'temperature':
    projected_quantity = np.load(pre + fix + '/T_' + fix + '.npy')
else:
    raise ValueError('Hate to break it to you champ \n \
                     but we don\'t have that quantity')    
# Specify
if what == 'density':
    cb_text = r'$\log_{10}$ Density [g/cm$^2$]'
    vmin = 0
    vmax = 7
elif what == 'temperature':
    cb_text = r'$\log_{10}$ Temperature [K]'
    vmin = 0
    vmax = 7
else:
    raise ValueError('Hate to break it to you champ \n \
                     but we don\'t have that quantity')
        
pro_q = np.nan_to_num(projected_quantity, nan = -1, neginf = -1)
pro_q = np.log10(pro_q)
pro_q = np.nan_to_num(pro_q, nan = 0, neginf= 0)

above = np.where(Z < 2, 1, 0)
below = np.where(Z > -2, 1, 0)
mask = np.multiply(above, below)
print(np.sum(mask))
# Mask midplane
X = X[mask > 0]
Y = Y[mask > 0]
pro_q = pro_q[mask > 0]

# Mask high T
high_T = np.where(pro_q > 7, 1, 0)
X_ht = X[high_T > 0]
Y_ht = Y[high_T > 0]
pro_q_ht = pro_q[high_T > 0]
#%%

step = 1
fig, ax = plt.subplots(1,1)
img = ax.scatter(X[::step] / Rt, Y[::step] / Rt, c = pro_q[::step], 
                 cmap = 'cet_fire', s = 0.01, vmin = 0, vmax = 9)
img2 = ax.scatter(X_ht[::step] / Rt, Y_ht[::step] / Rt, c = pro_q_ht[::step], 
                  cmap = 'cet_fire', s = 10, vmin = 0, vmax = 9, edgecolor = 'b')
ax.scatter(0,0, marker = 'x', c = 'k', s = 50)
cb = plt.colorbar(img)
cb.set_label(cb_text, fontsize = 14)
ax.set_xlabel(r' X/$R_T$ [R$_\odot$]', fontsize = 14)
ax.set_ylabel(r' Y/$R_T$ [R$_\odot$]', fontsize = 14)
ax.set_xlim(-40, 5)
ax.set_ylim(-20, 20)
ax.set_title('Midplane', fontsize = 16)
ax.plot(np.array(photo_x) / Rt, np.array(photo_y) / Rt, 
        marker = 'o', color = 'springgreen', linewidth = 3)
