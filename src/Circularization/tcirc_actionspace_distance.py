#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 14:17:30 2024

@author: konstantinos
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import colorcet
import src.Utilities.prelude as c


def tfb(Mbh, mstar = 0.5, rstar = 0.47):
    dE = -mstar/rstar * ((Mbh/mstar)**(1/3) + 1)
    a_tilde = - Mbh/(dE)
    return 2*np.pi*a_tilde**(3/2)/Mbh**(1/2)

pre = 'data/tcirc/'
Mbhs = [4, 5, 6]
fig, axs = plt.subplots(3,1, figsize = (6,6), 
                        tight_layout = True, sharex=True, sharey=True)
axs = axs.flatten()
for Mbh, ax in zip(Mbhs, axs):
    data = np.genfromtxt(f'{pre}meandists{Mbh}.csv', delimiter = ',').T
    days =  data[1]
    sorter = np.argsort(days)
    days = days[sorter]
    norm = np.max(data[2])
    dists = data[2] / norm
    dists = dists[sorter]
    
    dot = np.gradient(dists, days)
    tcirc = np.abs(dists/dot)

    ax.plot(days, dists, color = 'k', lw = 1, marker = 'o', markersize = 3)
    
    ax2 = ax.twinx()
    ax2.plot(days, tcirc, color = 'r', lw=1, ls = '--',
             marker = 'o', markersize = 3)
    
    ax2.set_ylim(1e-3,6e0)
    #ax.set_xlim(0.8)
    ax.set_yscale('log')
    ax2.set_yscale('log')

    ax.set_title(f'$M_\mathrm{{BH}} = 10^{Mbh} M_\odot$')
ax.set_ylabel('Distance / $d_\mathrm{max}$', fontsize = 14)
ax2.set_ylabel('$t_\mathrm{circ}$ [$t_\mathrm{FB}$]', fontsize = 14)
ax.set_xlabel('Time [$t_\mathrm{FB}$]', fontsize = 14)