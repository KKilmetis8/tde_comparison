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
check = '-fid'
t_fall = 40 * (Mbh/1e6)**(0.5)  # days EMR+20 p13

if m == 4 and check == '-fid':
    snapshots = [177, 178, 179, 180, 181, 
                 231, 232, 233, 234, 235, 
                 285, 286, 287, 288, 289, 
                 318, 319, 320, 321, 322] #np.arange(100, 322 + 1)
    days = [0.4974, 0.505, 0.515, 0.525, 0.5325, 
            0.995, 1.005, 1.0125, 1.0225, 1.0325,
            1.495, 1.5025, 1.5125, 1.5225, 1.53,
            1.8, 1.81, 1.82, 1.8275, 1.8375]
if m ==  4 and check == '-S60ComptonHires':
    snapshots = [210, 211, 212, 213, 214, 
                 234, 235, 236, 237, 238, 
                 269, 270, 271] #np.arange(210, 271 + 1)
    days = [0.7825, 0.79, 0.8, 0.81, 0.8175,
            1.0225, ]

for snap, day in zip(snapshots, days):
    if check == '-fid':
        pre = 'data/den4-fid'
    if check == '-S60ComptonHires':
        pre = 'data/den4-chr/'
    data = np.loadtxt(pre + 'denproj'+ str(m) + check + str(snap) +'.txt')
    x_radii = np.loadtxt(pre + 'xarray'+ str(m) + check + '.txt') #simulator units
    y_radii = np.loadtxt(pre + 'yarray'+ str(m) + check + '.txt') #simulator units

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
    
    ax.text(txt_x, txt_y, 'Time: ' + str(day) +  r' [t/t$_{fb}$]',
            color='white', 
            fontweight = 'bold', 
            fontname = 'Consolas',
            fontsize = 12)
    plt.savefig('Figs/dproj4-fid-' + str(day) + '.png')