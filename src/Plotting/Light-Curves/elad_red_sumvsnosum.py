#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 10:44:56 2025

@author: konstantinos
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col

import colorcet

import src.Utilities.prelude as c

def peak_finder(red, t, lim = 0.4):
    start = np.argmin( np.abs(t - lim) )
    red_g = np.nan_to_num(red[start:], nan = 0)
    t_g = t[start:]
    
    peak = np.argmax(red_g)
    return red[start + peak], t[start + peak]

def Leddington(M):
    return 1.26e38 * M
pre = 'data/red/'
Mbhs = [4, 5, 6]
Mbhs = [6]
cols = ['k', c.AEK, 'maroon']
extra = 'beta1S60n1.5Compton'
Mbh = 6
fig, ax = plt.subplots(1,1, figsize = (5,4), tight_layout = True, sharex=True)
data = np.genfromtxt(f'{pre}/red_sumthomp{Mbh}.csv', delimiter = ',').T
days = data[1]
sorter = np.argsort(days)

L = data[2] #/ (4 * np.pi)

peak4, peaktime4 = peak_finder(L[sorter], days[sorter])
# Plot    
ax.plot(days[sorter], L[sorter], color='maroon', lw = 0.5,
        marker = 'o', markersize = 2,  label = f'$10^{Mbh}$ $M_\odot$ ABS+SCA')

ax.plot(peaktime4, peak4, c = 'white', marker = 'X', markersize = 8, 
        markeredgecolor = 'maroon', markeredgewidth = 0.65, alpha = 0.75,
       )

data = np.genfromtxt(f'{pre}/red_richex{Mbh}.csv', delimiter = ',').T
days = data[1]
sorter = np.argsort(days)

L = data[2] / (4 * np.pi)

peak4, peaktime4 = peak_finder(L[sorter], days[sorter])
# Plot    
ax.plot(days[sorter], L[sorter], color='gray', lw = 0.5,
        marker = 'o', markersize = 2,  label = f'$10^{Mbh}$ $M_\odot$ ROSS')

ax.plot(peaktime4, peak4, c = 'white', marker = 'X', markersize = 8, 
        markeredgecolor = 'gray', markeredgewidth = 0.65, alpha = 0.75,
       )

ax.axhline(Leddington(10**Mbh), ls = '--', c = 'maroon')

# Text
# ax.text(0.9, Leddington(10**Mbh) * 0.1, f'$10^{Mbh}$ $M_\odot$', color = co,
#             fontsize = 14)
ax.text(1.3, Leddington(10**Mbh) * 1.2,'$L_\mathrm{Edd}$', color = co,
            fontsize = 14 )
ax.text(0.9, 4e41 * 10**(Mbh - 4), f'10$^{Mbh} M_\odot$', color = co, 
        fontsize = 14)
plt.legend()
# Make nice
ax.set_yscale('log')
ax.set_xlim(0.69)
ax.set_ylim(7e42, 3e44)
# ax.legend(ncols = 3)
ax.set_xlabel('Time $[t_\mathrm{FB}]$', fontsize = 16)
ax.set_ylabel('$L_\mathrm{FLD}$ [erg/s]', fontsize = 16)

