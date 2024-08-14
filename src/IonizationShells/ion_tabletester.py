#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 14:02:30 2024

@author: konstantinos
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import src.Utilities.prelude as c

Ts = np.logspace(3, 11, num=100)
Dens = np.logspace(-13, 2, num = 100)
xH = np.load('src/IonizationShells/xH.npy')
xHe1 = np.load('src/IonizationShells/xHe1.npy')
xHe2 = np.load('src/IonizationShells/xHe2.npy')

#%%
import scipy.interpolate as sci

fig, ax = plt.subplots(3,2, figsize = (4,4), sharex=True, sharey=True,
                       tight_layout = True)
ax[0,0].pcolormesh(np.log10(Ts), np.log10(Dens), xH.T)

hion = sci.RectBivariateSpline(Ts, Dens, xH)

Ts2 = np.logspace(3, 11, num = 300)
Dens2 = np.logspace(-13, 2, num = 300)

newH = np.zeros((len(Ts2),len(Dens2)))
for i, T in enumerate(Ts2):
    for j, D in enumerate(Dens2):
        newH[i][j] = hion(T, D)
        
ax[0,1].pcolormesh(np.log10(Ts2), np.log10(Dens2), newH.T)

#
ax[1,0].pcolormesh(np.log10(Ts), np.log10(Dens), xHe1.T)
heion1 = sci.RectBivariateSpline(Ts, Dens, xHe1)
newHe1 = np.zeros((len(Ts2),len(Dens2)))
for i, T in enumerate(Ts2):
    for j, D in enumerate(Dens2):
        newHe1[i][j] = heion1(T, D)
        
ax[1,1].pcolormesh(np.log10(Ts2), np.log10(Dens2), newHe1.T)
#
ax[2,0].pcolormesh(np.log10(Ts), np.log10(Dens), xHe2.T)
heion2 = sci.RectBivariateSpline(Ts, Dens, xHe2)
newHe2 = np.zeros((len(Ts2),len(Dens2)))
for i, T in enumerate(Ts2):
    for j, D in enumerate(Dens2):
        newHe2[i][j] = heion2(T, D)
        
im = ax[2,1].pcolormesh(np.log10(Ts2), np.log10(Dens2), newHe2.T)
cax = fig.add_axes([1, 0.08, 0.1, 0.8])
cb = fig.colorbar(im, cax)

# Labels
fig.text(0.55, 0, 'logT [K]', ha='center', fontsize = 14)
fig.text(-0.02, 0.5, r'log $\rho$ [g/cm$^3$]', va='center', rotation='vertical', fontsize = 14)
fig.suptitle('Ionization Tables | Tomida EoS', fontsize = 16)

# Text
fig.text(0.25, 0.75, 'H+ Table', c = 'k', fontsize = 14)
fig.text(0.75, 0.75, 'H+ Inter.', c = 'k', fontsize = 14)
fig.text(0.25, 0.5, 'He+ Table', c = 'k', fontsize = 14)
fig.text(0.75, 0.5, 'He+ Inter.', c = 'k', fontsize = 14)
fig.text(0.25, 0.2, 'He++ Table', c = 'k', fontsize = 14)
fig.text(0.7, 0.2, 'He++ Inter.', c = 'k', fontsize = 14)