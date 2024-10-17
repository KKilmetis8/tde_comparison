#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 15:50:40 2024

@author: konstantinos
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import colorcet

import src.Utilities.prelude

rstar = 0.47
mstar = 0.5
Mbh = '1e+06'
extra = 'beta1S60n1.5Compton'
extra2 = 'beta1S60n1.5ComptonRes20'
simname = f'R{rstar}M{mstar}BH{Mbh}{extra}' 
Mbh = 1e+06

simname2 = f'R{rstar}M{mstar}BH{Mbh}{extra2}' 

pre = 'data/ef82/'
ecc = np.loadtxt(f'{pre}ecc{simname}.txt')
days = np.loadtxt(f'{pre}eccdays{simname}.txt')
# ecc2 = np.loadtxt(f'{pre}ecc{simname2}.txt')
# days2 = np.loadtxt(f'{pre}eccdays{simname2}.txt')

Rt = rstar * (Mbh/mstar)**(1/3) # Msol = 1, Rsol = 1
apocenter = Rt * (Mbh/mstar)**(1/3)

radii_start = np.log10(0.4*Rt)
radii_stop = np.log10(apocenter) # apocenter
radii = np.logspace(radii_start, radii_stop, 1000) / apocenter

# # diff = np.abs(ecc[:len(ecc2)] - ecc2)
# # 19 - 19+len(days2) for HR to SHR
# q1 = 1-ecc[19:19+len(ecc2)]
# q2 = 1-ecc2
# ang_mom_deficit = np.abs((q1-q2)) / q1
#%%
fig, ax = plt.subplots(1,1, figsize = (4,4))

# img1 = ax.pcolormesh(radii, days, ang_mom_deficit, 
#                       norm = col.LogNorm(vmin = 1e-3, vmax = 1),
#                       cmap = 'cet_rainbow4')

img1 = ax.pcolormesh(radii, days, ecc, 
                      vmin = 0, vmax = 1, cmap = 'cet_rainbow4')
cb = fig.colorbar(img1)
#cb.set_label(r'Ang. Mom. Deficit $\frac{|(1-e_1) - (1-e_2)|}{ 1-e_1 }$ ', fontsize = 14, labelpad = 5)
cb.set_label('Eccentricity', fontsize = 14, labelpad = 5)
plt.xscale('log')
plt.ylim(0.12, 1.5)

plt.axvline(Rt/apocenter, c = 'k')
plt.axhline(0.843, c='hotpink')
plt.text(Rt/apocenter + 0.002, 0.3, '$R_\mathrm{T}$', 
         c = 'k', fontsize = 14)

plt.axvline(0.6 * Rt/apocenter, c = 'grey', ls = ':')
plt.text(0.6 * Rt/apocenter + 0.00001, 0.2, '$R_\mathrm{soft}$', 
         c = 'grey', fontsize = 14)
# Axis labels
fig.text(0.5, -0.01, r'r/R$_a$', ha='center', fontsize = 14)
fig.text(-0.02, 0.5, r' Time / Fallback time $\left[ t/t_{FB} \right]$', va='center', rotation='vertical', fontsize = 14)
ax.tick_params(axis = 'both', which = 'both', direction='in')
#ax.set_title(r'$10^6$ M$_\odot$')
plt.title('$ 10^6 M_\odot$')
#%%
import src.Utilities.prelude as c
fig, ax = plt.subplots(1,1, figsize = (4,4))

a = 25
b = 50
ci = 70
plt.plot(radii, np.abs(ang_mom_deficit[a]), 
         c = 'k', lw = 2, label = f'{days[a]:.2f} t/$t_{{FB}}$')
plt.plot(radii, np.abs(ang_mom_deficit[b]), ls = ':', 
         c = c.AEK, lw = 2, label = f'{days[b]:.2f} t/$t_{{FB}}$')
plt.plot(radii, np.abs(ang_mom_deficit[ci]), ls = ':',
         c = 'maroon', lw = 2, label = f'{days[ci]:.2f} t/$t_{{FB}}$')

plt.xscale('log')
plt.yscale('log')

# Rt
plt.axvline(Rt/apocenter, c = 'r', ls = '--')
plt.text(Rt/apocenter + 0.001, 5.95, '$R_\mathrm{T}$', 
         c = 'r', fontsize = 17)
plt.axvline(0.6 * Rt/apocenter, c = 'b', ls = '--')
plt.text(0.6 * Rt/apocenter + 0.001, 5.95, '$R_\mathrm{S}$', 
         c = 'b', fontsize = 17)
# Labels
plt.xlabel('Radius [r/$R_\mathrm{a}$]', fontsize = 14)
plt.ylabel(r'Ang. Mom. Deficit $\frac{|(1-e_1) - (1-e_2)|}{ 1-e_1 }$ ', fontsize = 13)
plt.legend(loc = 'lower right', ncol = 1)
plt.title('$10^4 M_\odot$ | HR vs SHR')