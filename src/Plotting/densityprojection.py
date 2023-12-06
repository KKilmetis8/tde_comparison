#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 06 

@author: paola
"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

import numpy as np

import matplotlib.pyplot as plt
import colorcet
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.figsize'] = [6, 4]
plt.rcParams['axes.facecolor']= 	'whitesmoke'

m = 4
Mbh = 10**m 
Rt =  Mbh**(1/3) # Msol = 1, Rsol = 1
apocenter = 2 * Rt * Mbh**(1/3)  # There is m_* hereeee

if m ==4:
    snapshots = [322]

for snap in snapshots:
    data = np.loadtxt('data/denproj'+ str(m) + '_' + str(snap) +'.txt')
    x_radii = np.loadtxt('data/xarray'+ str(m) + '.txt') #simulator units
    y_radii = np.loadtxt('data/xarray'+ str(m) + '.txt') #simulator units

    fig, ax = plt.subplots(1,1)
    den_plot = np.nan_to_num(data, nan = -1, neginf = -1)
    den_plot = np.log10(den_plot)
    den_plot = np.nan_to_num(den_plot, neginf= 0)


    ax.set_xlabel(r' X [$R_\odot$]', fontsize = 14)
    ax.set_ylabel(r' Y [R$_\odot$]', fontsize = 14)
    img = ax.pcolormesh(x_radii, y_radii, den_plot.T, cmap = 'cet_fire',
                        vmin = 0, vmax = 7)
    cb = plt.colorbar(img)
    cb.set_label(r'Density [g/cm$^2$]', fontsize = 14)
    ax.set_title('XY Projection', fontsize = 16)
    plt.show()