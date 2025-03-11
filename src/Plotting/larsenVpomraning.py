#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:21:20 2025

@author: konstantinos
"""

import numpy as np
import matplotlib.pyplot as plt
import src.Utilities.prelude as c 

def larsen(R):
    return 1/np.sqrt(9+R**2)

def levermore_pomraning(R):
    return 1/R * (1/np.tanh(R) - 1/R)

Rs = np.logspace(-3, 3)
larsens = [ larsen(R) for R in Rs ]
levermore_pomranings = [ levermore_pomraning(R) for R in Rs ]

plt.figure( figsize = (3,3))
plt.plot(Rs, larsens, c = 'k', 
         label = r'Larsen: $\left(3^2 + R^2 \right)^{-1/2}$')
plt.plot(Rs, levermore_pomranings, c = 'dodgerblue', ls = '--',
         label = r'Lev-Por: $\frac{1}{R} \left( \coth{R} - \frac{1}{R} \right)$')
plt.plot(Rs, 3*np.array(levermore_pomranings), c = 'coral', ls = '--',
         label = r'Wrong Lev-Por: $\frac{3}{R} \left( \coth{R} - \frac{1}{R} \right)$')
plt.text(2e-3, 2e-1, 'Opt. Thick')
plt.text(1.5e1, 2e-1, 'Opt. Thin')

plt.loglog()
plt.xlabel(r'$R = - \frac{|\nabla E|}{\alpha E}$, $\alpha = \kappa \rho$')
plt.ylabel(r'Flux Limiter $\lambda$')

plt.axhline(1/3, c = 'grey', label = '1/3', ls = ':')
plt.legend(frameon = 0, fontsize = 7)