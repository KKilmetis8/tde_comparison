#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 17:18:51 2024

@author: konstantinos
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import colorcet

from src.Plotting.circularizationtime import timer
import src.Utilities.prelude as c
from scipy.ndimage import uniform_filter1d 
pre = 'data/ef82/'
extra = 'beta1S60n1.5Compton'

def tfb(Mbh, mstar, rstar):
    return 40 * (Mbh/1e6)**(1/2) * mstar**(-1) * rstar**(3/2)
rstar = 0.47
mstar = 0.5
Mbhs = [10_000, 100_000, '1e+06']
#%%
tcs = []
days = []
for Mbh in Mbhs:
    simname = f'R{rstar}M{mstar}BH{Mbh}{extra}' 
    ecc = np.loadtxt(f'{pre}ecc{simname}.txt')
    day = np.loadtxt(f'{pre}eccdays{simname}.txt')
    mass = np.loadtxt(f'{pre}eccmass{simname}.txt')
    energy = np.loadtxt(f'{pre}eccenergy{simname}.txt')
    sma = np.loadtxt(f'{pre}eccsemimajoraxis{simname}.txt')
    
    Mbh = float(Mbh)
    Rt = rstar * (float(Mbh)/mstar)**(1/3)
    apocenter = Rt * (float(Mbh)/mstar)**(1/3)
    radii_start = np.log10(0.4*Rt)
    radii_stop = np.log10(apocenter) # apocenter
    radii = np.logspace(radii_start, radii_stop, 1000)#  / Rt
    
    rp = sma * (1-ecc)
    nick_E_circ = float(Mbh) / (4 * rp)
    angmom = np.sqrt(sma * float(Mbh) * (1 - ecc**2))
    egoal = - Mbh**2/(2*angmom**2)
    ecirc = np.zeros_like(energy) +  Mbh/(4*Rt)

    tc_nicke_none = timer(day, radii, q = energy, weights = None, Rt = Rt,
                          goal = egoal)
    days.append(day)
    tcs.append(tc_nicke_none)
    print(f'{Mbh:1.0e}')
    print(np.mean(tc_nicke_none[-10:]))
    print(np.mean(tc_nicke_none[-10:]) * tfb(Mbh, mstar, rstar))
#%%
plt.figure()
plt.plot(days[0], tcs[0], '-', c = 'k', label = '4')
plt.plot(days[1], tcs[1], '-', c = c.AEK, label = '5')
plt.plot(days[2], tcs[2], '-', c = 'maroon', label = '6')

plt.legend(bbox_to_anchor = [1, 0.5, 0.1, 0.1])
plt.yscale('log')
#plt.ylim(0, 10)
#plt.xlim(1)
plt.ylabel(r'$|t_\mathrm{circ}|$ [$t_\mathrm{FB}$]', fontsize = 13)
plt.xlabel(r't [$t_\mathrm{FB}$]', fontsize = 13)
plt.title(f'Circ. Time | Averaging 1-5 $R_\mathrm{{T}}$')
    