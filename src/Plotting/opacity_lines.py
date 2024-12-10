#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 12:51:57 2024

@author: konstantinos
"""

# The goal is to replicate Elad's script. No nice code, no nothing. A shit ton
# of comments though. 
import numpy as np
import matplotlib.pyplot as plt

import src.Utilities.prelude as c
from src.Opacity.linextrapolator import extrapolator_flipper


# Opacity Input
opac_path = f'src/Opacity/LTE_data/'
T_cool = np.loadtxt(f'{opac_path}/T.txt')
Rho_cool = np.loadtxt(f'{opac_path}/rho.txt')
plank = np.loadtxt(f'{opac_path}/planck.txt')
rossland = np.loadtxt(f'{opac_path}/ross.txt')
scattering = np.loadtxt(f'{opac_path}/scatter.txt')
sumof = np.log(np.exp(scattering) + np.exp(plank))

# Normals
Rho_n = np.exp(Rho_cool)
T_n = np.exp(T_cool)
ross_n = np.exp(rossland)
abs_n = np.exp(plank)
scattering_n = np.exp(scattering)
sumof_n = np.exp(sumof) 

what = 'rich'
T_cool2, Rho_cool2, rossland2 = extrapolator_flipper(T_cool, Rho_cool, rossland.T,
                                                     slope_length = 5, extrarows = 100,
                                                     what = what)
_, _, plank2 = extrapolator_flipper(T_cool, Rho_cool, plank.T, extrarows = 100,
                                    what = what)
_, _, scattering2 = extrapolator_flipper(T_cool, Rho_cool, scattering.T, extrarows = 100,
                                    what = what)
_, _, sumof2 = extrapolator_flipper(T_cool, Rho_cool, sumof.T, extrarows = 100,
                                    what = what)
T2_n = np.exp(T_cool2).T
Rho2_n = np.exp(Rho_cool2).T
ross2_n = np.exp(rossland2).T
abs2_n = np.exp(plank2).T
scattering2_n = np.exp(scattering2).T
sumof2_n = np.exp(sumof2).T
#%%
plt.figure()
plt.title('Within the table')
target_temperature = 1e7
ridx = np.argmin(np.abs(T_n - target_temperature))
plt.plot(Rho_n, ross_n[ridx], 'o-', c = 'k', lw = 0.75, markersize = 1,
         label = 'Ross')
# plt.plot(Rho_n, abs_n[ridx], 'o-',c = 'skyblue', lw = 0.75, markersize = 1,
#          label = 'Abs')
plt.plot(Rho_n, scattering_n[ridx] / Rho_n, 'o-', c = 'hotpink', lw = 0.75,  markersize = 1,
         label = 'Scatter')
# plt.plot(Rho_n, sumof_n[ridx], 'o--', c = c.c99, lw = 0.75,  markersize = 1,
#          label = 'Scatter + Abs')
plt.loglog()
plt.grid()
plt.axhline(0.34, c = 'k', ls = '--')
plt.xlabel('Density [g/cm3]')
plt.ylabel('Opacity ')
plt.legend(bbox_to_anchor = [1,0.8,0.1,0.1])
#%%
plt.figure()
plt.title('Outside the table')
ridx = np.argmin(np.abs(T2_n - target_temperature))
plt.plot(Rho2_n, ross2_n[ridx]  / Rho2_n, 'o-', c = 'k', lw = 0.75, markersize = 1,
         label = 'Ross')
plt.plot(Rho2_n, abs2_n[ridx] / Rho2_n, 'o-',c = 'skyblue', lw = 0.75, markersize = 1,
         label = 'Abs')
plt.plot(Rho2_n, scattering2_n[ridx]/Rho2_n, 'o-', c = 'hotpink', lw = 0.75,  markersize = 1,
          label = 'Scatter')
plt.axhline(0.34, c = 'k', ls = '--')
# plt.plot(Rho2_n, sumof2_n[ridx]/Rho2_n, 'o--', c = c.c99, lw = 0.75,  markersize = 1,
#          label = 'Scatter + Abs')
plt.axvline(np.min(Rho_n), ls = '--', c ='forestgreen')
plt.axvline(np.max(Rho_n), ls = '--', c ='forestgreen')
plt.xscale('log')
plt.loglog()
#plt.ylim(0, 0.5)
plt.grid()
plt.xlabel('Density [g/cm3]')
plt.ylabel('Opacity [cm2/g]')
plt.legend(bbox_to_anchor = [1,0.8,0.1,0.1])