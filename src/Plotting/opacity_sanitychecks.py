#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 16:25:15 2024

@author: konstantinos
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import colorcet

import src.Utilities.prelude as c

# Check by taking half of the thing
m = 4
fix = 164
data05 = np.genfromtxt(f'data/photosphere/sanity_half_{fix}_{m}.csv', delimiter = ',')
half_color = data05[-192:-1]
data1 = np.genfromtxt(f'data/photosphere/sanity_normal_{fix}_{m}.csv', delimiter = ',')
color = data1[-192:-1]
data2 = np.genfromtxt(f'data/photosphere/sanity_double_{fix}_{m}.csv', delimiter = ',')
double_color = data2[-192:-1]


fig, axs = plt.subplots(3,2, figsize = (8,6), dpi = 300, tight_layout = True)
axf = axs.flatten()
for i, ax in enumerate(axf):
    start = int( i * 192 / len(axf) )
    stop = int( (i+1) * 192 / len(axf) )
    x = np.arange(len(half_color))[start:stop]
    width = 2
    ax.plot(x, half_color[start:stop], 's', color=c.cyan, markersize = 5,
            label='Half', alpha = 1)
    ax.plot(x, color[start:stop], 'o', color='k', markersize = 4,
            label='Normal', alpha = 1)
    ax.plot(x, double_color[start:stop], '^', color=c.reddish, markersize = 3,
            label='Double', alpha = 1)

axs[-1,0].set_ylabel('Colorsphere Distance [$R_\odot$]')
axs[-1,0].set_xlabel('Observer no.')
fig.suptitle('Sanity check: Doubling/halving $\kappa_\mathrm{Planck}$')
axs[-1,0].legend()
#ax.set_ylim(-2, 500)
#%% Extrapolation

data05 = np.genfromtxt(f'data/photosphere/sanity_extra05_{fix}_{m}.csv', delimiter = ',')
half_color = data05[-193:-1]
data1 = np.genfromtxt(f'data/photosphere/sanity_extra1_{fix}_{m}.csv', delimiter = ',')
color = data1[-193:-1]
data2 = np.genfromtxt(f'data/photosphere/sanity_extra2_{fix}_{m}.csv', delimiter = ',')
double_color = data2[-193:-1]


fig, axs = plt.subplots(3,2, figsize = (8,6), dpi = 300, tight_layout = True)
axf = axs.flatten()
for i, ax in enumerate(axf):
    start = int( i * 192 / len(axf) )
    stop = int( (i+1) * 192 / len(axf) )
    x = np.arange(len(half_color))[start:stop]
    width = 2
    ax.plot(x, half_color[start:stop], 's', color=c.cyan, markersize = 5,
            label='Square Root', alpha = 1)
    ax.plot(x, color[start:stop], 'o', color='k', markersize = 4,
            label='Linear', alpha = 1)
    ax.plot(x, double_color[start:stop], '^', color=c.reddish, markersize = 3,
            label='Quadratic', alpha = 1)

axs[-1,0].set_ylabel('Colorsphere Distance [$R_\odot$]')
axs[-1,0].set_xlabel('Observer no.')
axs[-1,0].legend()
fig.suptitle('Sanity check: Extrapolate as not lin-in-log')

#%% Same for photo

data05 = np.genfromtxt(f'data/photosphere/sanity_extra05_{fix}_{m}.csv', delimiter = ',')
half_color = data05[5:5+192]
data1 = np.genfromtxt(f'data/photosphere/sanity_extra1_{fix}_{m}.csv', delimiter = ',')
color = data1[5:5+192]
data2 = np.genfromtxt(f'data/photosphere/sanity_extra2_{fix}_{m}.csv', delimiter = ',')
double_color = data2[5:5+192]


fig, axs = plt.subplots(3,2, figsize = (8,6), dpi = 300, tight_layout = True)
axf = axs.flatten()
for i, ax in enumerate(axf):
    start = int( i * 192 / len(axf) )
    stop = int( (i+1) * 192 / len(axf) )
    x = np.arange(len(half_color))[start:stop]
    width = 2
    ax.plot(x, half_color[start:stop], 's', color=c.cyan, markersize = 5,
            label='Square Root', alpha = 1)
    ax.plot(x, color[start:stop], 'o', color='k', markersize = 4,
            label='Linear', alpha = 1)
    ax.plot(x, double_color[start:stop], '^', color=c.reddish, markersize = 3,
            label='Quadratic', alpha = 1)

axs[-1,0].set_ylabel('Photosphere Distance [$R_\odot$]')
axs[-1,0].set_xlabel('Observer no.')
axs[-1,0].legend()
fig.suptitle('Sanity check: Extrapolate as not lin-in-log')