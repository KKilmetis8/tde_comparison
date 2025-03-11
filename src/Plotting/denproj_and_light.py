#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 14:55:59 2025

@author: konstantinos
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 16:55:12 2024

@author: konstantinos
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
from scipy.optimize import curve_fit
import colorcet
import src.Utilities.prelude as c

def peak_finder(red, t, lim = 0.4):
    start = np.argmin( np.abs(t - lim) )
    red_g = np.nan_to_num(red[start:], nan = 0)
    t_g = t[start:]
    
    peak = np.argmax(red_g)
    return red[start + peak], t[start + peak]
def tfb(m, mstar = 0.5, rstar = 0.47,):
    rstar *= c.Rsol_to_cm
    mstar *= c.Msol_to_g
    Mbh = 10**m * c.Msol_to_g
    return np.pi * np.sqrt(Mbh / (2 * c.Gcgs)) * rstar**(3/2) / mstar

fontsize = 15
fixes = [195, 227, 288, 302, 349]
fix = 349

fig = plt.figure(figsize = (7,3), constrained_layout=True)
gs = fig.add_gridspec(1, 3)
ax1 = fig.add_subplot(gs[0:2])
ax2 = fig.add_subplot(gs[2:])


pre = 'data/red/'
Mbh = 5
# Mbhs = [6]
extra = 'beta1S60n1.5Compton'
peaks = []
peaktimes = []
    #DeltaE = mstar/rstar * ( (Mbh/mstar)**(1/3) - 1 )
data = np.genfromtxt(f'{pre}/red_walljumper2{Mbh}.csv', delimiter = ',').T
days = data[1]
sorter = np.argsort(days)
days = days[sorter]    



L = data[2] #/ (4 * np.pi) # this is for the error
L = L[sorter]

mask = days > 0.5
days = days[mask]
L = L[mask]
this_idx = np.argmin(np.abs(fix - data[0][sorter][mask]))
this_time = days[this_idx]
peak4, peaktime4 = peak_finder(L, days)
peaks.append(peak4)
peaktimes.append(peaktime4)
# Light    

tfb_to_days = tfb(Mbh) / (60*60*24)
days *= tfb_to_days
peak4, peaktime4 = peak_finder(L, days)

ax2.plot(days[:this_idx], L[:this_idx], color='k', lw = 0.5,
        marker = 'o', markersize = 2,  label = f'10$^{Mbh}$ M$_\odot$')

if this_time * tfb_to_days > peaktime4:
    ax2.plot(peaktime4, peak4, c = 'white', marker = 'X', markersize = 8, 
            markeredgecolor = 'k', markeredgewidth = 0.65, alpha = 0.75,
           )
# Make nice
ax2.set_yscale('log')
ax2.set_xlim(4.8,13)
ax2.set_ylim(5e41, 1e43)
ax2.set_xlabel('Time [days]', fontsize = fontsize)
ax2.set_ylabel('$L_\mathrm{FLD}$ [erg/s]', fontsize = fontsize)

#-- Den proj
pre = 'data/denproj/paper/'
suf = 'beta1S60n1.5Compton'
f5 = fix

rstar = 0.47
mstar = 0.5
Rt5 = rstar * (1e5/mstar)**(1/3)
amin5 = Rt5 * (1e5/mstar)**(1/3)

step = 1
den5 = np.loadtxt(f'{pre}5normal{f5}.txt')[::step].T[::step].T
x5 = np.loadtxt(f'{pre}5normalx.txt')[::step]
y5 = np.loadtxt(f'{pre}5normaly.txt')[::step]

# Plot projection data
dmin = 0.1
dmax = 5
# try:
img = ax1.pcolormesh(x5/amin5, y5/amin5, den5.T, cmap = 'cet_fire',
                         vmin = dmin, vmax = dmax)
# except:
#     x5 = np.loadtxt(f'{pre}5normalx2.txt')[::step]
#     y5 = np.loadtxt(f'{pre}5normaly2.txt')[::step]
#     img = ax1.pcolormesh(x5/amin5, y5/amin5, den5.T, cmap = 'cet_fire',
#                              vmin = dmin, vmax = dmax)
ax1.text(0.05, 0.9, '$M_\mathrm{BH} = $10$^5$ M$_\odot$',
             fontsize = fontsize, c = 'white', transform = ax1.transAxes)
ax1.text(0.05, 0.05, f'{this_time * tfb_to_days:.2f} days',
             fontsize = fontsize, c = 'white', transform = ax1.transAxes)

ax1.set_xlabel(r'X $[\alpha_\mathrm{min}]$', fontsize = fontsize)
ax1.set_ylabel(r'Y $[\alpha_\mathrm{min}]$', fontsize = fontsize)
cb = fig.colorbar(img, fraction = 0.35, pad = 0.01)
cb.set_label('$\log_{10} (\Sigma) $ [g/cm$^2$]', fontsize = fontsize)
cb.ax.tick_params(labelsize = fontsize-4)
