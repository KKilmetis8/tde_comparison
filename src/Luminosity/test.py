#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 22:50:00 2023

@author: konstantinos
"""
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [6 , 6]

# Healpix
NSIDE = 16
NPIX = hp.nside2npix(NSIDE)
print(
    "Approximate resolution at NSIDE {} is {:.2} deg".format(
        NSIDE, hp.nside2resol(NSIDE, arcmin=True) / 60
    )
) 
m = np.arange(NPIX)
hp.mollview(m, title="Mollview image RING")
hp.graticule()

# Mine
thetas_hp = np.zeros(192)
phi_hp = np.zeros(192)

for i in range(0,192):
   thetas_hp[i], phi_hp[i] = hp.pix2ang(NSIDE, i)
   thetas_hp[i] -= np.pi/2
   phi_hp[i] -= np.pi

t_num = 7
p_num = 16
thetas = np.linspace(-np.pi/2, np.pi/2, num = t_num) 
phis = np.linspace(- np.pi , np.pi, num = p_num)


#%% Plot

fig, ax = plt.subplots(1,1, subplot_kw=dict(projection="mollweide"))
ax.scatter(phi_hp, thetas_hp, c = 'red', s = 25)
for theta in thetas:
    for phi in phis:
        ax.scatter(phi, theta, c = 'k', s=20, marker = 'h')
plt.grid(True)
