#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 22:50:00 2023

@author: konstantinos

Test how healpy divide the space in solid angles
"""
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [10 , 3]

NSIDE = 4
m = 6
fix = 1008
num = 5300 #700 for 844, 3000 for 925, 5360 for 1008
index_num = 2
see_observers = False
photosphere = True
flux = True

thetas = np.zeros(192)
phis = np.zeros(192)
observers = []
for i in range(0,192):
    thetas[i], phis[i] = hp.pix2ang(NSIDE, i)
    thetas[i] -= np.pi/2 # Enforce theta in -pi/2 to pi/2
    phis[i] -= np.pi # Enforce theta in -pi to pi
    observers.append( (thetas[i], phis[i]) )

# Plot
if see_observers:
    fig, ax = plt.subplots(1,1, subplot_kw=dict(projection="mollweide"))
    img = ax.scatter(phis, thetas, c = np.arange(192), s=20, marker = 'h')
    cbar = fig.colorbar(img)
    cbar.set_label('Observer', fontsize = 8)
    plt.grid(True)
    plt.savefig('Final plot/observers.png')
    plt.show()

if flux:
    flux = np.loadtxt('data/red/flux_m' + str(m) + '_fix' + str(fix) + '.txt')[index_num]/1e15
    fig, ax = plt.subplots(1,1, subplot_kw=dict(projection = "mollweide"))
    img = ax.scatter(phis, thetas, c = flux, cmap = plt.cm.coolwarm, s=15, marker = 'h', vmin = -5, vmax = 5)
    cbar = fig.colorbar(img)
    cbar.set_label(r'Flux [$10^{15}$ erg/$cm^2$s]', fontsize = 8)
    plt.grid(True)
    plt.savefig('Final plot/observers_flux' + str(fix) + '.png')
    plt.show()

if photosphere: 
    Mbh = 10**m
    Rt = Mbh**(1/3)
    Rref= 0.5 * Rt
    photo = np.loadtxt('data/red/photosphere' + str(fix) + '_num' + str(num) + '.txt')
    photo_to_plot = (photo/Rref) - 1
    fig, ax = plt.subplots(1,1, subplot_kw=dict(projection = "mollweide"))
    img = ax.scatter(phis, thetas, c = photo_to_plot, cmap = plt.cm.Paired, s = 15, marker = 'h', vmin = 0, vmax = 100)
    cbar = fig.colorbar(img)
    cbar.set_label(r'1-$\frac{R_{ph}}{R_{ph,ref}}$', fontsize = 8)
    plt.grid(True)
    plt.savefig('Final plot/observers_photo' + str(fix) + '.png')
    plt.show()
