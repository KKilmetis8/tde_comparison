#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 19:54:26 2024

@author: konstantinos
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt

import src.Utilities.prelude as c
m = [4, 5, 6]
mstar = 0.5
rstar = 0.47
cols = ['k', c.AEK, 'maroon']
fig, ax = plt.subplots(1,1, figsize=(4,3))

for i in range(len(m)):
    Mbh = 10**m[i]
    deltaE = 2 * mstar/rstar * ((Mbh/mstar)**(1/3) + 1)
    M_calli = np.load(f'data/tcirc/m_calli{m[i]}.npy')
    E_calli = np.load(f'data/tcirc/e_calli{m[i]}.npy')

    ax.plot(E_calli/deltaE, M_calli, c = cols[i], lw = 1.25, 
            label = f'10$^{m[i]}$ M$_\odot$')
    # norm = mstar / (2 * deltaE)
    # ax.text(2.5, norm*2, '$M_*/2\Delta E$')
    # ax.axhline(norm, c = 'k', ls = ':')

ax.axvline(-1, c = 'b', ls ='--')
# axs.axvline(0, c = 'white', ls =':')
ax.axvline(1, c = 'b', ls ='--')
ax.set_yscale('log')
ax.set_ylabel('dM/dE [code unit]')
ax.set_xlabel('Energy $[\Delta E]$')
ax.legend(bbox_to_anchor = (0.68, 0.4), frameon = False)