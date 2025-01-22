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


from src.Opacity.LTE_loader import T_opac_ex, Rho_opac_ex, rossland_ex, plank_ex, rossland_ex2, scattering_ex2, scattering_ex
when = 'new'

if when == 'old':
    scattering2 = scattering_ex2
    rossland2 = rossland_ex2
else:
    scattering2 = scattering_ex
    rossland2 = rossland_ex
T_cool2 = T_opac_ex
Rho_cool2 = Rho_opac_ex
plank2 = plank_ex

T2_n = np.exp(T_cool2).T
Rho2_n = np.exp(Rho_cool2).T
ross2_n = np.exp(rossland2)
abs2_n = np.exp(plank2)
scattering2_n = np.exp(scattering2)
#%%
X = 0.9082339738214822 # From table prescription
thompson = 0.2 * (1 + X)#%%
# plt.figure()
# plt.title('Within the table')
target_temperature = 1e5
# ridx = np.argmin(np.abs(T_n - target_temperature))
# plt.plot(Rho_n, ross_n[ridx], 'o-', c = 'k', lw = 0.75, markersize = 1,
#          label = 'Ross')
# plt.plot(Rho_n, abs_n[ridx], 'o-',c = 'skyblue', lw = 0.75, markersize = 1,
#          label = 'Abs')
# plt.plot(Rho_n, scattering_n[ridx], 'o-', c = 'hotpink', lw = 0.75,  markersize = 1,
#          label = 'Scatter')

# plt.loglog()
# plt.grid()
# plt.plot(Rho_n, thompson * Rho_n, c = 'k', ls = '--')
# plt.xlabel('Density [g/cm3]')
# plt.ylabel('Opacity Coeff [1/cm]')
# plt.legend(bbox_to_anchor = [1,0.8,0.1,0.1])

plt.figure()
plt.title(f'{when} - T = {target_temperature:.0e} K')
ridx = np.argmin(np.abs(T2_n - target_temperature))
plt.plot(Rho2_n, ross2_n[ridx] / Rho2_n , 'o-', c = 'blueviolet', 
         lw = 1, markersize = 3, label = 'Ross')
plt.plot(Rho2_n, abs2_n[ridx] / Rho2_n , 'o-',c = 'skyblue', 
         lw = 0.75, markersize = 1, label = 'Abs')
plt.plot(Rho2_n, scattering2_n[ridx] / Rho2_n  , 'o-', c = 'hotpink',
         lw = 0.75,  markersize = 1, label = 'Scatter')
plt.plot(Rho2_n, thompson * np.ones(len(Rho2_n)), c = 'k', 
         ls = '--', zorder = 5)
plt.text(1e-20, thompson * 2, 'Thompson', c='k')
# plt.plot(Rho2_n, sumof2_n[ridx]/Rho2_n, 'o--', c = c.c99, lw = 0.75,  markersize = 1,
#          label = 'Scatter + Abs')
plt.axvline(np.min(Rho_n), ls = '--', c ='forestgreen', label = 'Table limits')
plt.axvline(np.max(Rho_n), ls = '--', c ='forestgreen')
plt.xscale('log')
plt.yscale('log')
# plt.loglog()
#plt.ylim(0, 0.5)
plt.grid()
plt.xlabel('Density [g/cm$^3$]')
plt.ylabel('Opacity [cm$^2$/g]')
plt.legend(bbox_to_anchor = [1,0.8,0.1,0.1])
#%% #%% Den
target_density = 1e-10
plt.figure()
ridx = np.argmin(np.abs(Rho2_n - target_density))
kramers =  1e22 * T2_n**(-3.5)
plt.plot(T2_n, ross2_n.T[ridx]/ target_density, 'o-',c = 'blueviolet', lw = 1, markersize = 3,
         label = 'Ross')
plt.plot(T2_n, abs2_n.T[ridx] / target_density, 'o-',c = 'skyblue', lw = 0.75, markersize = 1,
         label = 'Abs')
plt.plot(T2_n, scattering2_n.T[ridx] / target_density , 'o-',c = 'hotpink', lw = 0.75, markersize = 1,
         label = 'scattering')
plt.plot(T2_n[230:], kramers[230:] , ':',c = 'slateblue', lw = 0.75, markersize = 1,)
plt.plot(T2_n, thompson  * np.ones(len(T2_n)), c = 'k', ls = '--')

#plt.text(5e8, 5e-12, 'Kramers', c='slateblue', rotation = -65)
#plt.axhline(thompson, c = 'k', ls = ':')
plt.text(1e8, thompson*1e1, 'Thompson', c='k')
plt.loglog()
plt.axvline(np.max(T_n), c = 'forestgreen', ls = '--')
plt.axvline(np.min(T_n), c = 'forestgreen', ls = '--', alpha = 0.1)
plt.title(rf'{when} - $\rho$ = {target_density:.0e} g/cm$^3$')
plt.ylim(1e-5, 1e10)
plt.xlabel('Temperature [K]')
plt.ylabel('Opacity [cm$^2$/g]')
plt.legend(loc = 'lower center', fontsize = 6, ncols = 3)
#%% Many densities many plots
fig, axs  = plt.subplots(3,3, figsize = (10, 10), sharex = True, sharey = True)
target_densities = [1e-9, 1e-10, np.min(Rho_n), 1e-11, 1e-12, 1e-13, 1e-15, 1e-16, 1e-17]
for target_density, ax in zip(target_densities, axs.flatten()):
    if target_density > np.min(Rho_n):
        txt = 'In table'
    else:
        txt = 'Out of table'
    ridx = np.argmin(np.abs(Rho2_n - target_density))
    kramers =  1e22 * T2_n**(-3.5)
    ax.plot(T2_n, ross2_n.T[ridx]/ target_density, 'o-',c = 'blueviolet', lw = 1, markersize = 3,
             label = 'Ross')
    ax.plot(T2_n, abs2_n.T[ridx] / target_density, 'o-',c = 'skyblue', lw = 0.75, markersize = 1,
             label = 'Abs')
    ax.plot(T2_n, scattering2_n.T[ridx] / target_density , 'o-',c = 'hotpink', lw = 0.75, markersize = 1,
             label = 'scattering')
    ax.plot(T2_n[230:], kramers[230:] , ':',c = 'slateblue', lw = 0.75, markersize = 1,)
    ax.plot(T2_n, thompson  * np.ones(len(T2_n)), c = 'k', ls = '--')

    #plt.text(5e8, 5e-12, 'Kramers', c='slateblue', rotation = -65)
    #plt.axhline(thompson, c = 'k', ls = ':')
    ax.text(2e2, thompson*1e1, 'Thompson', c='k')
    ax.loglog()
    ax.axvline(np.max(T_n), c = 'forestgreen', ls = '--')
    ax.axvline(np.min(T_n), c = 'forestgreen', ls = '--', alpha = 0.1)
    ax.set_title(rf'{txt} - $\rho$ = {target_density:.0e} g/cm$^3$')
    ax.set_ylim(1e-5, 1e10)
    ax.set_xlim(1e2, 1e5)
    ax.set_xlabel('Temperature [K]')
    ax.set_ylabel('Opacity [cm$^2$/g]')
plt.legend(loc = 'lower center', fontsize = 6, ncols = 3)

#%% Many densities, one plot
plt.figure(figsize = (3,3))
target_densities = [Rho_n[4],  Rho_n[0], 1e-11, 1e-12, 1e-13, 1e-14, Rho_n[25]]
cols = [c.c91, c.c92, c.c93, c.c94, c.c95, c.c96, c.c97, c.c98, c.c99]
for target_density, col in zip(target_densities, cols):
    if target_density < np.min(Rho_n):
        ls = '--'
    else:
        ls = '-'
    ridx = np.argmin(np.abs(Rho2_n - target_density))
    plt.plot(T2_n, abs2_n.T[ridx] / target_density,c = col, ls = ls, 
            lw = 0.75, markersize = 1,
             label = f'{target_density:.2e}')
plt.yscale('log')
plt.legend(bbox_to_anchor = [1,0.25,0.5,0.5])
plt.axvline(np.max(T_n), c = 'forestgreen', ls = '--')
plt.axvline(np.min(T_n), c = 'forestgreen', ls = '--', alpha = 0.1)
plt.ylim(1e0, 1e5)
plt.xlim(5e3, 1e4)
plt.xlabel('Temperature [K]')
plt.ylabel('Opacity [cm$^2$/g]')
