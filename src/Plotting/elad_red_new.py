#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 16:55:12 2024

@author: konstantinos
"""

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

def peak_finder(red, t, lim = 0.45):
    start = np.argmin( np.abs(t - lim) )
    print(start)
    red_g = np.nan_to_num(red[start:])
    t_g = t[start:]
    
    peak = np.argmax(red_g)
    return red[start + peak], t[start + peak]

def Leddington(M):
    return 1.26e38 * M
pre = 'data/red/'
Mbhs = [4, 5, 6]
cols = ['k', c.AEK, 'maroon']
extra = 'beta1S60n1.5Compton'

fig, ax = plt.subplots(1,1, figsize = (5,4), tight_layout = True, sharex=True)
for Mbh, co in zip(Mbhs, cols):
    #DeltaE = mstar/rstar * ( (Mbh/mstar)**(1/3) - 1 )
    data = np.genfromtxt(f'{pre}eladred{Mbh}.csv', delimiter = ',').T
    days = data[1]
    L = data[2] / (4*np.pi) # CAREFUL
    data = np.genfromtxt(f'{pre}eladred{Mbh}_d19.csv', delimiter = ',').T
    days2 =  data[1]
    L2 = data[2] / (4*np.pi)
    peak4, peaktime4 = peak_finder(L, days)
    # Plot    
    ax.scatter(days, L, color=co, lw = 2)

    ax.plot(peaktime4, peak4, c = 'white', marker = 'X', markersize = 15, 
            markeredgecolor = co, markeredgewidth = 2)
    ax.axhline(Leddington(10**Mbh), ls = '--', c = co)

# Make nice
ax.set_yscale('log')
ax.set_xlim(0)
ax.set_ylim(1e41)

ax.set_xlabel('Time $[t_\mathrm{FB}]$', fontsize = 14)
ax.set_ylabel('$L_\mathrm{bol}$ [erg/s]', fontsize = 14)

