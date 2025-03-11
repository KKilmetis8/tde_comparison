#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 10:11:46 2025

@author: konstantinos
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import colorcet
import matplotlib.patheffects as pe 
import src.Utilities.prelude as c

pre = 'data/ef82/'
rstar = 0.47
mstar = 0.5
Mbhs = [10_000, 100_000, '1e+06']
extra = 'beta1S60n1.5Compton'

ytickmaxs = [1.4, 1.4, 1.4]
fig, axs = plt.subplots(1,3, figsize = (7,3), dpi = 400, tight_layout = True, sharey = True)
labelfontsize = 13

for Mbh, ax, ytickmax in zip(Mbhs, axs.flatten(), ytickmaxs):
    # Load
    simname = f'R{rstar}M{mstar}BH{Mbh}{extra}' 
    days = np.loadtxt(f'{pre}eccdays{simname}.txt')
    ecc = np.loadtxt(f'{pre}ecc{simname}.txt')
    
    # TDE specific
    Mbh = float(Mbh)
    Rt = rstar * (Mbh/mstar)**(1/3) # Msol = 1, Rsol = 1
    apocenter = Rt * (Mbh/mstar)**(1/3)
    radii_start = np.log10(0.4*Rt)
    radii_stop = np.log10(apocenter) # apocenter
    radii = np.logspace(radii_start, radii_stop, 1000) / apocenter
    step = 4
    img1 = ax.pcolormesh(radii[::step], days, ecc.T[::step].T, 
                          vmin = 0.2, vmax = 1, cmap = 'cet_CET_L6_r',
                          shading = 'nearest',)
    if Mbh == 1e6:
        cb = fig.colorbar(img1, aspect = 10)
    
    ax.axvline(Rt/apocenter, c = c.AEK)
    ax.axvline(0.6 * Rt/apocenter, c = 'hotpink', ls = '--')
    ax.set_xscale('log')
    bhtext = f'10$^{int(np.log10(Mbh))} \mathbf{{M}}_\odot$'
    # ax.set_title(bhtext, fontsize = labelfontsize)
    ax.text(0.6, 0.87, bhtext , 
            fontsize = labelfontsize, color = 'whitesmoke', transform = ax.transAxes,)
            # bbox=dict(facecolor='white', edgecolor='none', alpha = 0.15))
            #path_effects=[pe.withStroke(linewidth=2, foreground="k")])
            
    # Ticks
    yticks = np.linspace(0.2, ytickmax, 5)
    ylabels = [f'{x:1.1f}' for x in yticks]
    ax.set_yticks(yticks, labels=ylabels)
   
ax.set_ylim(0.15, 1.43)
axs[0].text(4e-2, 0.35, '$R_\mathrm{T}$', c = 'goldenrod', fontsize = labelfontsize)
axs[0].text(4e-2, 0.45, '$R_\mathrm{smoothing}$',  c = 'crimson', fontsize = labelfontsize)

axs[0].set_ylabel('Time $[t/t_\mathrm{FB}]$', fontsize = labelfontsize)
cb.set_label('Eccentricity', fontsize = labelfontsize, labelpad = 5)
axs[1].set_xlabel(r'Radial Coordinate [$r/ \alpha_\mathrm{min}$]', fontsize = labelfontsize)
plt.savefig('paperplots/eccevo.eps', dpi = 200, bbox_inches = 'tight')

