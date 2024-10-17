#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:17:38 2024

@author: konstantinos
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import colorcet

import src.Utilities.prelude as c
pre = 'data/energyhist/'
rstar = 0.47
mstar = 0.5
Mbhs = [4, 5, 6]
extra = 'beta1S60n1.5Compton'

fig, axs = plt.subplots(3,1, figsize = (5,5), tight_layout = True, sharex=True)
for Mbh, ax in zip(Mbhs, axs):
    #DeltaE = mstar/rstar * ( (Mbh/mstar)**(1/3) - 1 )
    energy = np.genfromtxt(f'{pre}ehist{Mbh}.csv', delimiter = ',').T
    nanmask = energy[2] != np.NaN
    days = energy[0][nanmask] 
    orb = energy[1][nanmask] #/ DeltaE
    ie = energy[2][nanmask] #/ DeltaE
    rad = energy[3][nanmask] #/ DeltaE
    tot = orb + ie + rad
    orb /= tot
    ie /= tot
    rad /= tot
    tot /= tot
    
    # Plot    
    ax.plot(days, orb, color=c.cyan)
    ax.plot(days, ie, color=c.reddish)
    ax.plot(days, rad, color=c.AEK)
    ax.plot(days, tot, color='k', ls ='--')
    
    # Make nice
    ax.set_title(f'$M_\mathrm{{BH}}$ $10^{Mbh} M_\odot$')
    # Scale
    ax.set_ylim(1e-2, 2)
    ax.set_yscale('log')
    if Mbh == Mbhs[0]:
        ax.set_xlim(0.7, days[-1])
    if Mbh == Mbhs[-1]:
        ax.set_xlabel('Time $[t_\mathrm{FB}]$', fontsize = 14)
        ax.set_ylabel('Energy/Total Energy', fontsize = 14)
        
        # Legend
        ax.plot([], [], color=c.cyan, label = 'Orbital')
        ax.plot([], [], color=c.reddish, label = 'Internal')
        ax.plot([], [], color=c.AEK, label = 'Radation')
        ax.plot([], [], color='k', ls ='--', label = 'Total')
        ax.legend()
#%%
fig, axs = plt.subplots(1,1, figsize = (5,4), tight_layout = True, sharex=True)
colors = ['k', c.AEK, 'maroon']
for Mbh, col in zip(Mbhs, colors):
    #DeltaE = mstar/rstar * ( (Mbh/mstar)**(1/3) - 1 )
    energy = np.genfromtxt(f'{pre}ehist{Mbh}.csv', delimiter = ',').T
    nanmask = energy[2] != np.NaN
    days = energy[0][nanmask] 
    orb = energy[1][nanmask] #/ DeltaE
    ie = energy[2][nanmask] #/ DeltaE
    rad = energy[3][nanmask] #/ DeltaE
    ratio = rad/orb
    
    # Plot    
    axs.plot(days, ratio, color=col, label = str(Mbh))
    

    # Scale
    axs.set_ylim(1e-2, 2)
    axs.set_yscale('log')
    if Mbh == Mbhs[0]:
        axs.set_xlim(0.7, days[-1])
    if Mbh == Mbhs[-1]:
        axs.set_xlabel('Time $[t_\mathrm{FB}]$', fontsize = 14)
        axs.set_ylabel('Rad/Orb', fontsize = 14)
        
axs.legend(loc = 'lower right')