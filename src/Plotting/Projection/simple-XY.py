#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 12:11:48 2024

@author: konstantinos
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import colorcet
from tqdm import tqdm
import matplotlib.patches as mp
import src.Utilities.prelude

rstar = 0.47
mstar = 0.5
Mbh = '1e+06'
#fixes = np.arange(101, 172+1) # 4s
#fixes = np.arange(80, 216+1) # 4h
#fixes = np.arange(180, 365+1) # 6
fixes = np.arange(80, 348+1) # 4
fixes = [351]
#fixes = np.arange(132, 365+1)
extra = 'beta1S60n1.5Compton'
simname = f'R{rstar}M{mstar}BH{Mbh}{extra}' 
times = np.loadtxt(f'data/ef8/eccdays{simname}.txt')
Mbh = float(Mbh)
what = 'Diss' # den
bar = 'Energy Dissipation [erg/s cm$^2$]' # whats plotted on the cb bar
                            # Density [g/cm$^2$]
pre = f'data/denproj/{simname}/'
if what == 'den':
    vmin = 1
    vmax = 5
if what == 'Diss':
    vmin = 0
    vmax = 100
# plt.ioff()


for i, fix in tqdm(enumerate(fixes)):
    den = np.loadtxt(f'{pre}{what}proj{simname}{fix}.txt')
    x = np.loadtxt(f'{pre}xarray{simname}.txt')
    y = np.loadtxt(f'{pre}yarray{simname}.txt')
    
    Rt = rstar * (Mbh/mstar)**(1/3) # Msol = 1, Rsol = 1
    apocenter = Rt * (Mbh/mstar)**(1/3)
    plt.figure()
    fig, ax = plt.subplots(1,1, figsize = (12,4))
    
    img1 = ax.pcolormesh(x, y, np.log10(den.T), 
                         cmap = 'cet_fire')
    circle = mp.Circle((0,0), Rt, color = 'b', fill = False, lw = 2)
    ax.add_patch(circle)
    cb = fig.colorbar(img1)
    cb.set_label(f'Log Column {bar}', fontsize = 13, labelpad = 5)
    plt.scatter(0,0, c='k', marker = 'X', s = 20, edgecolor = 'white')
    plt.title(f'$M_* = 0.5$ $M_{{\mathrm{{BH}}}} = 10^6 M_\odot$')# {times[i]:.2f} $t_{{\mathrm{{FB}}}}$')
    plt.xlabel('X $[R_\odot]$')
    plt.ylabel('Y $[R_\odot]$')
    #plt.savefig(f'/home/konstantinos/denproj/{simname}/{fix}.png')
    #plt.close()
    

