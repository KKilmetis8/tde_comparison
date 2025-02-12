#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 12:05:18 2025

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
    return red[start + peak], t[start + peak], peak

def tfb(m, mstar = 0.5, rstar = 0.47,):
    rstar *= c.Rsol_to_cm
    mstar *= c.Msol_to_g
    Mbh = 10**m * c.Msol_to_g
    return np.pi * np.sqrt(Mbh / (2 * c.Gcgs)) * rstar**(3/2) / mstar

def exponential_increase(t, L0, tau):
    return L0 * np.exp(t / tau)
def exponential_decay(t, L0, tau):
    return L0 * np.exp(-t / tau)


# Fit the model
pre = 'data/red/'
Mbhs = [4, 5, 6]
# Mbhs = [6]
cols = ['k', c.AEK, 'maroon']
extra = 'beta1S60n1.5Compton'
fig, ax = plt.subplots(2,2, figsize = (4,3), tight_layout = True, sharey=True)
for Mbh, co in zip(Mbhs, cols):
    data = np.genfromtxt(f'{pre}/red_richex{Mbh}.csv', delimiter = ',').T
    days = data[1]
    sorter = np.argsort(days)
    days = days[sorter]    
    
    L = data[2] / (4 * np.pi) # this is for the error
    L = L[sorter]
    
    mask = days > 0.69
    days = days[mask]
    # L = np.log10(L[mask])
    L = L[mask]
    days = days[~np.isnan(L)]
    L = L[~np.isnan(L)]
    _, _, peakidx = peak_finder(L, days)
    params, _ = curve_fit(exponential_increase, days[:peakidx], 
                                   L[:peakidx], p0=[1e43, 2])
    params2, _ = curve_fit(exponential_decay, days[peakidx:], 
                                   L[peakidx:], p0=[1e43, 1])
    # plt.figure()
    # plt.plot(days[peakidx:], L[peakidx:], c = 'k')
    # fit = [ exponential_decay(day, params2[0], params2[1]) for day in days[peakidx:]]
     
    # plt.plot(days[peakidx:], fit, c = 'r', ls = '--')
    tfb_to_days = tfb(Mbh) / (60*60*24)
    # days *= tfb_to_days
    ax[0,0].plot(params[1], 10**Mbh, 'h', 
               markeredgecolor = 'k', color = co)
    ax[0,1].plot(params[1] * tfb_to_days,  10**Mbh, 'h', 
               markeredgecolor = 'k', color = co)
    ax[1,0].plot(params2[1], 10**Mbh, 'h', 
               markeredgecolor = 'k', color = co)
    ax[1,1].plot(params2[1] * tfb_to_days,  10**Mbh, 'h', 
               markeredgecolor = 'k', color = co)
    
    
# Make nice
ax[0,0].set_yscale('log')
ax[0,0].set_ylim(5e3, 1.5e6)
# ax[0].set_xlim(0.69)
# ax[0].set_ylim(1e41, 7e44)
# ax[1].legend(frameon = False)
# # ax.legend(ncols = 3)
ax[0,0].set_xlabel('e-folding rise time $[t_\mathrm{FB}]$', fontsize = 10)
ax[0,1].set_xlabel('e-folding rise time [days]', fontsize = 10)
ax[1,0].set_xlabel('e-folding decay time $[t_\mathrm{FB}]$', fontsize = 10)
ax[1,1].set_xlabel('e-folding decay time [days]', fontsize = 10)
ax[0,0].set_ylabel('log$M_\mathrm{BH}$ [M$_\odot$]', fontsize = 10)
ax[1,0].set_ylabel('log$M_\mathrm{BH}$ [M$_\odot$]', fontsize = 10)

# plt.savefig('paperplots/lightcurves.pdf', dpi = 300)

