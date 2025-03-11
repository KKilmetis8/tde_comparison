#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 15:26:40 2025

@author: konstantinos
"""

import numpy as np
import matplotlib.pyplot as plt
import src.Utilities.prelude as c

def Rt(Mbh, mstar = 1, rstar = 1):
    return rstar * (Mbh/mstar)**(1/3) 

def rg(Mbh):
    rg = 2*c.Gcgs * Mbh * c.Msol_to_g / c.c**2
    return  rg / c.Rsol_to_cm

def rg_spin(Mbh):
    rg = 2*c.Gcgs * Mbh * c.Msol_to_g / c.c**2
    return  rg / c.Rsol_to_cm



Mbhs = np.logspace(3,10)
Rts = Rt(Mbhs)
rgs = rg(Mbhs)
rgs_spin = rg_spin(Mbhs)


plt.plot(Mbhs, rgs, c = 'k', lw = 2)
plt.plot(Mbhs, Rts, c = 'dodgerblue', lw = 2)
plt.plot(1.2e8, 5e2, marker = 'h', c = c.AEK, markeredgecolor = 'dodgerblue',
         markersize = 8, markeredgewidth = 1.25)
plt.text(1e6, 1, '$r_g \propto M_\mathrm{BH}$', c = 'k', fontsize = 14)
plt.text(1e3, 2e2, '$R_\mathrm{T} \propto M_\mathrm{BH}^{1/3}$', c = 'dodgerblue', fontsize = 14)
plt.text(2e8, 1e2, '$M_\mathrm{Hills}$', c = 'darkgoldenrod', fontsize = 14)

plt.xlabel('$M_\mathrm{BH}$ $[M_\odot]$')
plt.ylabel('Length scales $[R_\odot]$')


plt.loglog()