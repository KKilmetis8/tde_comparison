#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 16:07:36 2024

@author: konstantinos
"""

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
Mbh = 10000
fixes4 = [164, 237, 313]
fixes5 = [208, 268, 365]
titles = [r'0.5 $t_\mathrm{FB}$','1 $t_\mathrm{FB}$', '1.5 $t_\mathrm{FB}$']
extra = 'beta1S60n1.5Compton'
simname4 = f'R{rstar}M{mstar}BH{Mbh}{extra}' 
simname5 = f'R{rstar}M{mstar}BH{Mbh}0{extra}' 

def plotter(ax, simname, fix, txt, title, Mbh):
    pre = f'data/denproj/{simname}/'
    den = np.loadtxt(f'{pre}denproj{simname}{fix}.txt')
    x = np.loadtxt(f'{pre}xarray{simname}.txt')
    y = np.loadtxt(f'{pre}yarray{simname}.txt')
    
    Rt = rstar * (Mbh/mstar)**(1/3) # Msol = 1, Rsol = 1
    Ra = Rt * (Mbh/mstar)**(1/3)
    
    img1 = ax.pcolormesh(x/Ra, y/Ra, np.log10(den.T), vmin = 1, vmax = 5,
                         cmap = 'cet_fire')
    circle = mp.Circle((0,0), Rt/Ra, color = 'b', fill = False, lw = 2)
    ax.add_patch(circle)
    
    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(-0.05, 0.05)
    ax.text(0.87, 0.1, txt, c = 'white', transform = ax.transAxes, fontsize = 20)
    return img1


fig, ax = plt.subplots(2,1, figsize = (8,4))
plt.xlabel('X $[r/R_a]$', fontsize = 20)
fig.text(0.04,0.4, 'Y $[r/R_a]$', fontsize = 20, transform = fig.transFigure,
         rotation = 90)
when = 2
fig.suptitle(titles[when], fontsize = 20, y = 0.96)
_ = plotter(ax[0], simname4, fixes4[when], r'$10^4 M_\odot$', titles[when], 1e4)
img1 = plotter(ax[1], simname5, fixes5[when], r'$10^5 M_\odot$', titles[when], 1e5)
cax = fig.add_axes([0.93,0.12,0.04,0.76])
cb = fig.colorbar(img1, cax = cax)
cb.set_label('Log Column Density [g/cm$^2$]', fontsize = 20, labelpad = 7,)

