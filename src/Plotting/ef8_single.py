#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 15:50:40 2024

@author: konstantinos
"""

import numpy as np
import matplotlib.pyplot as plt
import colorcet

import src.Utilities.prelude

rstar = 0.47
mstar = 0.5
Mbh = 10000
extra = 'beta1S60n1.5Compton'
simname = f'R{rstar}M{mstar}BH{Mbh}{extra}' 

pre = 'data/ef8/'
ecc = np.loadtxt(f'{pre}ecc{simname}.txt')
days = np.loadtxt(f'{pre}eccdays{simname}.txt')

Rt = rstar * (Mbh/mstar)**(1/3) # Msol = 1, Rsol = 1
apocenter = Rt * (Mbh/mstar)**(1/3)

radii_start = np.log10(0.4*Rt)
radii_stop = np.log10(apocenter) # apocenter
radii = np.logspace(radii_start, radii_stop, 200) / apocenter
# radii4 = np.linspace(0.2*2*Rt4, apocenter, 100) 
 
####
fig, ax = plt.subplots(1,1, figsize = (4,4))

img1 = ax.pcolormesh(radii, days, ecc,
                     cmap = 'cet_rainbow4', vmin = 0, vmax = 1)


cax = fig.add_axes([0.99, 0.065, 0.02, 0.86])
fig.colorbar(img1)#, cax=cax)
# ax[1].set_xlim(4e-3, 1)
#ax[1].set_ylim(days4chr[0], days4chr[-1])

ax[0].set_xscale('log')
ax[1].set_xscale('log')

# Distance text 
ionx = 1.06
iony = 0.4
#txt1.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='k')])
txt1 = fig.text(ionx, iony, 'Eccentricity', fontsize = 14,
		    color='k', fontfamily = 'monospace', rotation = 270)

# Axis labels
fig.text(0.5, -0.01, r'r/R$_a$', ha='center', fontsize = 14)
fig.text(-0.02, 0.5, r' Time / Fallback time $\left[ t/t_{FB} \right]$', va='center', rotation='vertical', fontsize = 14)
ax[0].tick_params(axis = 'both', which = 'both', direction='in')
#ax[1].tick_params(axis = 'both', which = 'both', direction='in')
ax.set_title(r'$10^4$ Fiducial M$_\odot$ - Fiducial ')
#ax[1].set_title(r'$10^4$  M$_\odot$ - Compton HiRes')
# fig.suptitle('Mass Weigh Eccentricity', fontsize = 17)
plt.savefig('Final plot/ecc05.png')
plt.show()