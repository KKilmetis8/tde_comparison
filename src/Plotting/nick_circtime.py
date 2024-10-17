#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 10:46:10 2024

@author: konstantinos
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import colorcet
import src.Utilities.prelude as c
from scipy.ndimage import uniform_filter1d 

rstar = 0.47
mstar = 0.5
Mbh = '1e+06'
extra = 'beta1S60n1.5Compton'
simname = f'R{rstar}M{mstar}BH{Mbh}{extra}' 
pre = 'data/ef82/'
ecc = np.loadtxt(f'{pre}ecc{simname}.txt')
days = np.loadtxt(f'{pre}eccdays{simname}.txt')
mass = np.loadtxt(f'{pre}eccmass{simname}.txt')
energy = np.loadtxt(f'{pre}eccenergy{simname}.txt')
sma = np.loadtxt(f'{pre}eccsemimajoraxis{simname}.txt')
rp = sma * (1-ecc)
nick_E_circ = float(Mbh) / (4 * rp)
angmom = np.sqrt(sma * float(Mbh) * (1 - ecc**2))
Rt = rstar * (float(Mbh)/mstar)**(1/3) # Msol = 1, Rsol = 1
apocenter = Rt * (float(Mbh)/mstar)**(1/3)
nick_C = float(Mbh) / (4 * Rt)
radii_start = np.log10(0.4*Rt)
radii_stop = np.log10(apocenter) # apocenter
radii = np.logspace(radii_start, radii_stop, 1000)#  / Rt

def timer2(time, radii, c, q, weights, Rt):
    qdot = np.full_like(q, np.nan)
    t_circ = np.full_like(q, np.nan)
    
    for i in range(len(radii)):
        q_on_an_r = q.T[i]
        # c_on_an_r = c.T[i]
        mask = ~np.isnan(q_on_an_r)
        qdot_temp = np.gradient(q_on_an_r[mask], time[mask])
        qdot.T[i][mask] = qdot_temp

        t_circ_temp = np.divide(c, qdot_temp)  
        t_circ.T[i][mask] = np.abs(t_circ_temp)
    
    minR = 1 * Rt
    minidx = int(np.argmin(np.abs(radii - minR)))
    maxR = 6 * Rt
    maxidx = int(np.argmin(np.abs(radii - maxR)))
    avg_range = np.arange(minidx, maxidx)
    t_circ_w = np.zeros(len(time))
    
    if type(weights) == type(None):
        for i in range(len(time)):
            for j, r in enumerate(avg_range):
                t_circ_w[i] += t_circ[i][r]  # i is time, r is radius
            t_circ_w[i] = np.divide(t_circ_w[i], len(avg_range))
    else:
        for i in range(len(time)):
            for j, r in enumerate(avg_range):
                t_circ_w[i] += t_circ[i][r] * weights[i][r] # i is time, r is radius
    return t_circ_w

tc_nidea_mass = timer2(days, radii, c = nick_C, q = nick_E_circ, 
                      weights = mass, Rt = Rt)
tc_nidea_energy = timer2(days, radii, c = nick_C, q = nick_E_circ, 
                      weights = np.abs(energy), Rt = Rt)
tc_nidea_none = timer2(days, radii, c = nick_C, q = nick_E_circ, 
                      weights = None, Rt = Rt)

#%%
plt.axvline(1.108, c='r')
plt.plot(days, tc_nidea_mass, 'h', c = 'k', markersize = 3,
          alpha = 1, label = r'$E/\dot{E}_{circ}$ - Mass')
plt.plot(days, tc_nidea_energy, '^', c = c.reddish, markersize = 3,
          alpha = 0.75, label = r'$E/\dot{E}_{circ}$ - Energy')
plt.plot(days, tc_nidea_none, 's', c = c.cyan, markersize = 3,
          alpha = 0.5, label = r'$E/\dot{E}_{circ}$ - None')

plt.legend(bbox_to_anchor = [1, 0.5, 0.1, 0.1])
plt.yscale('log')
plt.ylabel(r'$|t_\mathrm{circ}|$ [$t_\mathrm{FB}$]', fontsize = 13)
plt.xlabel(r't [$t_\mathrm{FB}$]', fontsize = 13)
Mbht = int(np.log10(Mbh))
plt.title(f'$10^{Mbht} M_\odot$ | Circ. Time | Averaging 1-6 $R_\mathrm{{T}}$')