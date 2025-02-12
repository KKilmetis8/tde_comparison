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
m = int(np.log10(float(Mbh)))
Rt = rstar * (float(Mbh)/mstar)**(1/3) # Msol = 1, Rsol = 1
apo = Rt * (float(Mbh)/mstar)**(1/3)
#fixes = np.arange(101, 172+1) # 4s
#fixes = np.arange(80, 216+1) # 4h
#fixes = np.arange(180, 365+1) # 6
fixes = np.arange(80, 348+1) # 4
fixes = [444]
#fixes = np.arange(132, 365+1)
extra = 'beta1S60n1.5Compton'
simname = f'R{rstar}M{mstar}BH{Mbh}{extra}' 
times = np.loadtxt(f'data/ef8/eccdays{simname}.txt')
quantity = 'Den' # den
where = 'local' # local or alice
if where == 'local':
    pre = f'{m}/'
if where == 'alice':
    pre = f'data/denproj/{simname}/'
if quantity == 'Den':
    vmin = -10
    vmax = -7
    bar = 'Density [g/cm$^3$]' # whats plotted on the cb bar
if quantity == 'Diss':
    vmin = -9
    vmax = -6
    bar = 'Energy Dissipation [erg/s cm$^2$]' # whats plotted on the cb bar

# plt.ioff()


for i, fix in tqdm(enumerate(fixes)):
    if where == 'alice':
        den = np.loadtxt(f'{pre}{quantity}proj{simname}{fix}.txt')
        x = np.loadtxt(f'{pre}xarray{simname}.txt')
        y = np.loadtxt(f'{pre}yarray{simname}.txt')
        
        plt.figure()
        fig, ax = plt.subplots(1,1, figsize = (8,6))
        img1 = ax.pcolormesh(x/apo, y/apo, np.log10(den.T), 
                             cmap = 'cet_fire')
        
    if where == 'local':
        step = 100
        den = np.load(f'{pre}{fix}/{quantity}_{fix}.npy')
        x = np.load(f'{pre}{fix}/CMx_{fix}.npy')
        y = np.load(f'{pre}{fix}/CMy_{fix}.npy')
        z = np.load(f'{pre}{fix}/CMz_{fix}.npy')
        
        zup = z<2
        zdown = z>-2
        zmask = zup * zdown
        x = x[zmask][::step]
        y = y[zmask][::step]
        den = den[zmask][::step]

        fig, ax = plt.subplots(1,1, figsize = (6,5))
        img1 = ax.scatter(x/apo, y/apo, c = np.log10(np.abs(den)), s = 0.1,
                             cmap = 'cet_fire', vmin = vmin, vmax = vmax)
        
    circle = mp.Circle((0,0), Rt/apo, color = 'b', fill = False, lw = 2)
    ax.add_patch(circle)
    cb = fig.colorbar(img1)
    cb.set_label(f'Log  {bar}', fontsize = 13, labelpad = 5)
    plt.scatter(0,0, c='k', marker = 'X', s = 20, edgecolor = 'white')
    plt.title(f'$M_* = 0.5$ $M_{{\mathrm{{BH}}}} = 10^{m} M_\odot$')# {times[i]:.2f} $t_{{\mathrm{{FB}}}}$')
    plt.xlabel(r'X $[\alpha_\mathrm{min}]$')
    plt.ylabel(r'Y $[\alpha_\mathrm{min}]$')
    plt.xlim(-0.05, 0.05)
    plt.ylim(-0.05, 0.05)
    #plt.savefig(f'/home/konstantinos/denproj/{simname}/{fix}.png')
    #plt.close()
    

