#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 09:50:41 2025

@author: konstantinos
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 15:50:40 2024

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

ytickmaxs = [1.53, 1.53,]
fig, axs = plt.subplots(2,1, figsize = (4,7), tight_layout = True)
labelfontsize = 13
Mbh = 100_000
paths = ['3','2']
for pa, ax, ytickmax in zip(paths, axs.flatten(), ytickmaxs):
    Mbh = int(Mbh)

    # Load
    simname = f'R{rstar}M{mstar}BH{Mbh}{extra}' 
    days = np.loadtxt(f'{pre}eccdays{simname}.txt')
    ecc = np.loadtxt(f'{pre}ecc{pa}{simname}.txt')
    
    # TDE specific
    Mbh = float(Mbh)
    Rt = rstar * (Mbh/mstar)**(1/3) # Msol = 1, Rsol = 1
    apocenter = Rt * (Mbh/mstar)**(1/3)
    radii_start = np.log10(0.4*Rt)
    radii_stop = np.log10(apocenter) # apocenter
    title_text = 'Runge-Lenz vector (old)'
    if pa == '2' or pa == '3':
        radii_stop = np.log10(2*apocenter) # apocenter
        title_text = r'$e = \sqrt{ 1 + \frac{2 \epsilon l^2}{G^2M_\mathrm{BH}^2}}$ (new)'
    radii = np.logspace(radii_start, radii_stop, 1000) / apocenter
    step = 1
    img1 = ax.pcolormesh(radii[::step], days, ecc.T[::step].T, 
                          vmin = 0.65, vmax = 1, cmap = 'cet_CET_L6_r',
                          shading = 'nearest',)
    cb = fig.colorbar(img1, aspect = 10)
    
    ax.axvline(Rt/apocenter, c = c.AEK)
    ax.axvline(0.6 * Rt/apocenter, c = 'hotpink', ls = '--')
    ax.set_xscale('log')
    bhtext = f'10$^{int(np.log10(Mbh))} \mathbf{{M}}_\odot$'
    # ax.set_title(bhtext, fontsize = labelfontsize)
    ax.text(0.7, 0.85, bhtext , 
            fontsize = labelfontsize, color = 'whitesmoke', transform = ax.transAxes,)
            # bbox=dict(facecolor='white', edgecolor='none', alpha = 0.15))
            #path_effects=[pe.withStroke(linewidth=2, foreground="k")])
    ax.set_xlim(7e-3, 1)
    ax.set_title(title_text, fontsize = labelfontsize+2)
    # Ticks
    yticks = np.linspace(0.2, ytickmax, 5)
    ylabels = [f'{x:1.1f}' for x in yticks]
    ax.set_yticks(yticks, labels=ylabels)
    
axs[1].text(9e-3, 0.2, '$R_\mathrm{T}$', c = 'goldenrod', fontsize = labelfontsize)
axs[1].text(9e-3, 0.4, '$R_\mathrm{smoothing}$',  c = 'crimson', fontsize = labelfontsize)

axs[1].set_ylabel('Time $[t/t_\mathrm{FB}]$', fontsize = labelfontsize)
cb.set_label('Eccentricity', fontsize = labelfontsize, labelpad = 5)
axs[1].set_xlabel(r'Radial Coordinate [$r/ \alpha_\mathrm{min}$]', fontsize = labelfontsize)
# plt.savefig('paperplots/eccevo.eps', dpi = 200, bbox_inches = 'tight')
#%%
fig, ax = plt.subplots(1,1, figsize = (4,4), tight_layout = True)
ecc2 = np.loadtxt(f'{pre}ecc2{simname}.txt')
ecc3 = np.loadtxt(f'{pre}ecc3{simname}.txt')
y = 1 - ecc2/ecc3
img1 = ax.pcolormesh(radii[::step], days, y.T[::step].T, 
                      vmin = -1, vmax = 1, cmap = 'cet_CET_L6_r',
                      shading = 'nearest',)
cb = fig.colorbar(img1, aspect = 10)
ax.axvline(Rt/apocenter, c = c.AEK)
ax.axvline(0.6 * Rt/apocenter, c = 'hotpink', ls = '--')
ax.set_xscale('log')
bhtext = f'10$^{int(np.log10(Mbh))} \mathbf{{M}}_\odot$'
# ax.set_title(bhtext, fontsize = labelfontsize)
ax.text(0.7, 0.85, bhtext , 
        fontsize = labelfontsize, color = 'whitesmoke', transform = ax.transAxes,)
ax.text(9e-3, 0.2, '$R_\mathrm{T}$', c = 'goldenrod', fontsize = labelfontsize)
ax.text(9e-3, 0.4, '$R_\mathrm{smoothing}$',  c = 'crimson', fontsize = labelfontsize)

ax.set_ylabel('Time $[t/t_\mathrm{FB}]$', fontsize = labelfontsize)
cb.set_label('1 - ($e_\mathrm{new}/e_\mathrm{old}$)', fontsize = labelfontsize, labelpad = 5)
ax.set_xlabel(r'Radial Coordinate [$r/ \alpha_\mathrm{min}$]', fontsize = labelfontsize)