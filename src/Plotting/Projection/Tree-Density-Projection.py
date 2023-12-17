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
from src.Luminosity.select_path import select_snap
import colorcet
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.figsize'] = [6, 4]
plt.rcParams['axes.facecolor']= 	'whitesmoke'

# Choose simulation
m = 4
check = 'fid'

Mbh = 10**m 
Rt =  Mbh**(1/3) # Msol = 1, Rsol = 1
apocenter = 2 * Rt * Mbh**(1/3)  # There is m_* hereeee
t_fall = 40 * (Mbh/1e6)**(0.5)  # days EMR+20 p13

#snapshots, days = select_snap(m, check)
snapshots = np.arange(210, 322 + 1)
days = np.loadtxt('data/red/alicered4fid_days.txt')

for snap, day in zip(snapshots, days):
    pre = 'data/denproj/' + str(m)
    sim = str(m) + '-' + check 
    data = np.loadtxt(pre + '/denproj' + sim + str(snap) + '.txt')
    x_radii = np.loadtxt(pre + '/xarray' + sim + '.txt') #simulator units
    y_radii = np.loadtxt(pre + '/yarray' + sim + '.txt') #simulator units

    fig, ax = plt.subplots(1,1)
    den_plot = np.nan_to_num(data, nan = -1, neginf = -1)
    den_plot = np.log10(den_plot)
    den_plot = np.nan_to_num(den_plot, neginf= 0)

    ax.set_xlabel(r' X [$x/R_a$]', fontsize = 14)
    ax.set_ylabel(r' Y [$y/R_a$]', fontsize = 14)
    img = ax.pcolormesh(x_radii/apocenter, y_radii/apocenter, den_plot.T, 
                        cmap = 'cet_fire',
                        vmin = 0, vmax = 6)
    cb = plt.colorbar(img)
    cb.set_label(r'Density [g/cm$^2$]', fontsize = 14)
    ax.set_title('XY Projection', fontsize = 16)
    
    txt_x = (x_radii[0] + 50) / apocenter
    txt_y = (y_radii[0] + 50) / apocenter
    
    ax.text(txt_x, txt_y, 'Time: ' + str(round(day,5)) +  r' [t/t$_{fb}$]',
            color='white', 
            fontweight = 'bold', 
            fontname = 'Consolas',
            fontsize = 12)
    plt.savefig(f'Figs/denproj/{m}/denproj{sim}{snap}.png')
    # plt.show()