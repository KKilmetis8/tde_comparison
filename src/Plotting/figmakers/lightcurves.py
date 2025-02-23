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
def powerlaw(x, A, B):
    return A * x**B

import src.Utilities.prelude as c

def peak_finder(red, t, lim = 0.4):
    start = np.argmin( np.abs(t - lim) )
    red_g = np.nan_to_num(red[start:], nan = 0)
    t_g = t[start:]
    
    peak = np.argmax(red_g)
    return red[start + peak], t[start + peak]

def Leddington(M):
    return 1.26e38 * M

def tfb(m, mstar = 0.5, rstar = 0.47,):
    rstar *= c.Rsol_to_cm
    mstar *= c.Msol_to_g
    Mbh = 10**m * c.Msol_to_g
    return np.pi * np.sqrt(Mbh / (2 * c.Gcgs)) * rstar**(3/2) / mstar

pre = 'data/red/'
Mbhs = [4,]# 5, 6]
# Mbhs = [6]
cols = ['k', c.AEK, 'maroon']
extra = 'beta1S60n1.5Compton'
peaks = []
peaktimes = []
fig, ax = plt.subplots(1,2, figsize = (5,4), tight_layout = True, sharey=True)
for Mbh, co in zip(Mbhs, cols):
    #DeltaE = mstar/rstar * ( (Mbh/mstar)**(1/3) - 1 )
    data = np.genfromtxt(f'{pre}/red_walljumper{Mbh}.csv', delimiter = ',').T
    days = data[1]
    sorter = np.argsort(days)
    days = days[sorter]    
    
    L = data[2]# / (4 * np.pi) # this is for the error
    L = L[sorter]

    mask = days > 0.69
    days = days[mask]
    L = L[mask]
    peak4, peaktime4 = peak_finder(L, days)
    peaks.append(peak4)
    peaktimes.append(peaktime4)
    # Plot    
    ax[0].plot(days, L, color=co, lw = 0.5,
            marker = 'o', markersize = 2,  label = f'10$^{Mbh}$ M$_\odot$')

    ax[0].plot(peaktime4, peak4, c = 'white', marker = 'X', markersize = 8, 
            markeredgecolor = co, markeredgewidth = 0.65, alpha = 0.75,
           )
    ax[0].axhline(Leddington(10**Mbh), ls = '--', c = co)
    
    tfb_to_days = tfb(Mbh) / (60*60*24)
    days *= tfb_to_days
    peak4, peaktime4 = peak_finder(L, days)
    
    ax[1].plot(days, L, color=co, lw = 0.5,
            marker = 'o', markersize = 2,  label = f'10$^{Mbh}$ M$_\odot$')

    ax[1].plot(peaktime4, peak4, c = 'white', marker = 'X', markersize = 8, 
            markeredgecolor = co, markeredgewidth = 0.65, alpha = 0.75,
           )
    ax[1].axhline(Leddington(10**Mbh), ls = '--', c = co)
    
    # Text
    # ax[0].text(0.9, Leddington(10**Mbh) * 0.1, f'$10^{Mbh}$ $M_\odot$', color = co,
    #             fontsize = 14)
    ax[0].text(1.5, Leddington(10**Mbh) * 1.2,'$L_\mathrm{Edd}$', color = co,
                fontsize = 14 )
    # ax.text(0.9, 4e41 * 10**(Mbh - 4), f'10$^{Mbh} M_\odot$', color = co, 
    #         fontsize = 14)


# Make nice
ax[0].set_yscale('log')
ax[0].set_xlim(0.69)
ax[0].set_ylim(1e41, 7e44)
ax[1].legend(frameon = False)
# ax.legend(ncols = 3)
ax[0].set_xlabel('Time $[t_\mathrm{FB}]$', fontsize = 16)
ax[1].set_xlabel('Time [days]', fontsize = 16)
ax[0].set_ylabel('$L_\mathrm{FLD}$ [erg/s]', fontsize = 16)

# peaks fit
peak_mass = curve_fit(powerlaw, 10**np.array(Mbhs), peaks, p0=[1e42, 1])[0]
fit = [ powerlaw(10**Mbh, peak_mass[0], peak_mass[1]) for Mbh in Mbhs ]
plt.figure(figsize = (3,3))
for Mbh, peak, co in zip(Mbhs, peaks, cols):
    plt.plot(10**np.array(Mbh), peak, 'h', c = co, markeredgecolor = 'k',
             markersize = 5)
exp = f'{peak_mass[0]:.2e}'[-2:]
flt = f'{peak_mass[0]:.2e}'[:4]

label = f'$L_\\mathrm{{peak}} =$ {flt}$\\times 10^{{{exp}}} M_\\mathrm{{BH}}^{{{peak_mass[1]:.2f}}}$'

plt.plot(10**np.array(Mbhs), fit, ls = '--', c = c.cyan, label = label)
plt.xlabel('$M_\mathrm{BH}$ [M$_\odot$]')
plt.legend(frameon = False, fontsize = 9)
plt.ylabel('$L_\mathrm{peak}$ [erg/s]')
plt.loglog()
# plt.savefig('paperplots/lightcurves.pdf', dpi = 300)

