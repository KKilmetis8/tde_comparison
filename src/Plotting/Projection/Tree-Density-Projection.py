#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 06 

@author: paola
"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

import numpy as np
import os
import matplotlib.pyplot as plt
import src.Utilities.selectors as s
import colorcet
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.figsize'] = [10, 4]
plt.rcParams['axes.facecolor']= 'whitesmoke'

# Choose simulation
m = 5
opac_kind = 'LTE'
check = 'fid'
mstar = 0.5
if mstar == 0.5:
    star = 'half'
else:
    star = ''
rstar = 0.47
beta = 1

Mbh = 10**m 
Rt =  rstar * (Mbh/mstar)**(1/3) 
apocenter = 2 * Rt * (Mbh/mstar)**(1/3) 

# snapshots, days = s.select_snap(m, mstar, rstar, check)
days = [1.5]
snapshots = [263]

for snap, day in zip(snapshots, days):
    sim = f'{m}{star}-{check}'
    pre = f'data/denproj/{sim}'
    file = f'{pre}/denproj{sim}{snap}.txt'
    x_radii = np.loadtxt(f'{pre}/xarray{sim}.txt') #simulator units
    y_radii = np.loadtxt(f'{pre}/yarray{sim}.txt') #simulator units
    if os.path.exists(file):
        data = np.loadtxt(file) #simulator units

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
        #plt.savefig(f'Figs/denproj/{m}/denproj{sim}0{snap}.png')
        plt.show()