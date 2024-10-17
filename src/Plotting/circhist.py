#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 16:05:52 2024

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
radii_start = np.log10(0.4*Rt)
radii_stop = np.log10(apocenter) # apocenter
radii = np.logspace(radii_start, radii_stop, 1000)#  / Rt
#%%

fig, ax = plt.subplots(5, 4, figsize=(14,14), sharex = True, 
                       tight_layout = True, sharey = True)
axflat = ax.ravel()
for i, eidx in enumerate(np.arange(146, 224, 4)):
    # Actually plotting
    axflat[i].hist(ecc[eidx][:-2], bins = 10,  weights = mass[eidx][:-2],
                   density = True,
                   color = 'k', ec = 'white')
    axflat[i].text(0.1, 0.9, f'{days[eidx]:.2f} t/tfb', fontsize = 16,
                   transform = axflat[i].transAxes)
    # Scales & Labels
axflat[i].set_xlabel(r'Eccentricity')
axflat[i].set_ylabel('\%')
