#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 12:38:03 2024

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
Mbh = 100000
extra = 'beta1S60n1.5Compton'
simname = f'R{rstar}M{mstar}BH{Mbh}{extra}' 
pre = 'data/ef82/'
ecc = np.loadtxt(f'{pre}ecc{simname}.txt')
days = np.loadtxt(f'{pre}eccdays{simname}.txt')
mass = np.loadtxt(f'{pre}eccmass{simname}.txt')
energy = np.loadtxt(f'{pre}eccenergy{simname}.txt')
sma = np.loadtxt(f'{pre}eccsemimajoraxis{simname}.txt')
Mbh = float(Mbh)
rp = sma * (1-ecc)
angmom = np.sqrt(sma * float(Mbh) * (1 - ecc**2))
egoal = - Mbh**2/(2*angmom**2)
Rt = rstar * (float(Mbh)/mstar)**(1/3) # Msol = 1, Rsol = 1
ecirc = np.zeros_like(energy) +  Mbh/(4*Rt)

apocenter = Rt * (float(Mbh)/mstar)**(1/3)
radii_start = np.log10(0.4*Rt)
radii_stop = np.log10(apocenter) # apocenter
radii = np.logspace(radii_start, radii_stop, 1000)#  / Rt
#%%

fig, axs = plt.subplots(4,4, figsize=(14,14), sharex=True, sharey=True)
axf = axs.ravel()
day_plot = np.linspace(0, len(days)-1, 16)
for i, d in enumerate(day_plot):
    d = int(d)
    img = axf[i].scatter(np.abs(energy[d][2:-1]), angmom[d][2:-1], c = mass[d][2:-1],
                         cmap = 'cet_CET_D7',
                         norm = colors.LogNorm(vmin = 1e-6, vmax = 1e-2))
    axf[i].text(0.05, 0.05, f'{days[d]:.2f} $t_\mathrm{{FB}}$', 
                transform = axf[i].transAxes, fontsize = 20)
    axf[i].set_xscale('log')
    axf[i].set_yscale('log')
axf[i].set_xlim(2e2, 1e9)
axf[i].set_ylim(2e3, 1.5e5)
fig.colorbar(img, cax = fig.add_axes([0.93,0.12,0.04,0.765]))#, transfrom = fig.transFigure)
fig.text(0.5, 0.08, 'Orbital Energy', fontsize = 25)
fig.text(0.07, 0.4, 'Angular Momentum', rotation = 90, fontsize = 25)

fig, axs = plt.subplots(4,4, figsize=(14,14), sharex=True, sharey=True)
axf = axs.ravel()
day_plot = np.linspace(0, len(days)-1, 16)
for i, d in enumerate(day_plot):
    d = int(d)
    img = axf[i].scatter(np.abs(energy[d][2:-1]), angmom[d][2:-1], c = ecc[d][2:-1],
                         cmap = 'cet_rainbow4', vmin = 0.25, vmax = 1)
    axf[i].text(0.05, 0.05, f'{days[d]:.2f} $t_\mathrm{{FB}}$', 
                transform = axf[i].transAxes, fontsize = 20)
    axf[i].set_xscale('log')
    axf[i].set_yscale('log')
fig.colorbar(img, cax = fig.add_axes([0.93,0.12,0.04,0.765]))#, transfrom = fig.transFigure)
fig.text(0.5, 0.08, 'Orbital Energy', fontsize = 25)
fig.text(0.07, 0.4, 'Angular Momentum', rotation = 90, fontsize = 25)
axf[i].set_xlim(2e2, 1e9)
axf[i].set_ylim(2e3, 1.5e5)

fig, axs = plt.subplots(4,4, figsize=(14,14), sharex=True, sharey=True)
axf = axs.ravel()
day_plot = np.linspace(0, len(days)-1, 16)
for i, d in enumerate(day_plot):
    d = int(d)
    img = axf[i].scatter(np.abs(energy[d][2:-1]), angmom[d][2:-1], c = radii[2:-1],
                         cmap = 'plasma')
    axf[i].text(0.05, 0.05, f'{days[d]:.2f} $t_\mathrm{{FB}}$', 
                transform = axf[i].transAxes, fontsize = 20)
    axf[i].set_xscale('log')
    axf[i].set_yscale('log')
fig.colorbar(img, cax = fig.add_axes([0.93,0.12,0.04,0.765]))#, transfrom = fig.transFigure)
fig.text(0.5, 0.08, 'Orbital Energy', fontsize = 25)
fig.text(0.07, 0.4, 'Angular Momentum', rotation = 90, fontsize = 25)
axf[i].set_xlim(2e2, 1e9)
axf[i].set_ylim(2e3, 1.5e5)
#%%
fig, axs = plt.subplots(4,4, figsize=(14,14), sharex=True, sharey=True)
axf = axs.ravel()
day_plot = np.linspace(140, 160, 16)
for i, d in enumerate(day_plot):
    d = int(d)
    img = axf[i].scatter(np.abs(energy[d][:-1]), angmom[d][:-1], c = ecc[d][:-1],
                         cmap = 'cet_rainbow4', vmin = 0.25, vmax = 1)
    axf[i].text(0.05, 0.05, f'{days[d]:.2f} $t_\mathrm{{FB}}$', 
                transform = axf[i].transAxes, fontsize = 20)
    axf[i].set_xscale('log')
    axf[i].set_yscale('log')
fig.colorbar(img, cax = fig.add_axes([0.93,0.12,0.04,0.765]))#, transfrom = fig.transFigure)
fig.text(0.5, 0.08, 'Orbital Energy', fontsize = 25)
fig.text(0.07, 0.4, 'Angular Momentum', rotation = 90, fontsize = 25)
axf[i].set_xlim(2e2, 1e9)
axf[i].set_ylim(2e3, 1.5e5)


fig, axs = plt.subplots(4,4, figsize=(14,14), sharex=True, sharey=True)
axf = axs.ravel()
day_plot = np.linspace(0, len(days)-1, 16)
for i, d in enumerate(day_plot):
    d = int(d)
    img = axf[i].scatter(np.abs(energy[d][2:-1]), angmom[d][2:-1], c = mass[d][2:-1],
                         cmap = 'cet_CET_D7',
                         norm = colors.LogNorm(vmin = 1e-6, vmax = 1e-2))
    axf[i].text(0.05, 0.05, f'{days[d]:.2f} $t_\mathrm{{FB}}$', 
                transform = axf[i].transAxes, fontsize = 20)
    axf[i].set_xscale('log')
    axf[i].set_yscale('log')
axf[i].set_xlim(2e2, 1e9)
axf[i].set_ylim(2e3, 1.5e5)
fig.colorbar(img, cax = fig.add_axes([0.93,0.12,0.04,0.765]))#, transfrom = fig.transFigure)
fig.text(0.5, 0.08, 'Orbital Energy', fontsize = 25)
fig.text(0.07, 0.4, 'Angular Momentum', rotation = 90, fontsize = 25)