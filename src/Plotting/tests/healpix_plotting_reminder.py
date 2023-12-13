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

# Healpix plot
NPIX = hp.nside2npix(NSIDE)
print(
    "Approximate resolution at NSIDE {} is {:.2} deg".format(
        NSIDE, hp.nside2resol(NSIDE, arcmin=True) / 60
    )
) 
m = np.arange(NPIX)
hp.orthview(m, title="Mollview image RING")
hp.graticule()

# Healpix and my plots
thetas_hp = np.zeros(192)
phis_hp = np.zeros(192)

for i in range(0,192):
   thetas_hp[i], phis_hp[i] = hp.pix2ang(NSIDE, i)
   # thetas_hp[i] -= np.pi/2
   # phis_hp[i] -= np.pi
   #print('theta:', thetas_hp[i], ',')
   #print('phi:', phis_hp[i] , '\n')
print('theta number: ', thetas_hp.shape, '\n')
print('theta unique number: ', np.unique(thetas_hp).shape, '\n')
print('phi number: ', phis_hp.shape, '\n')
print('phi unique number: ', np.unique(phis_hp).shape, '\n')
theta_num = 7
phi_num = 16
thetas = np.linspace(-np.pi/2, np.pi/2, num = theta_num) 
phis = np.linspace(- np.pi , np.pi, num = phi_num)


#%% Plot
fig, ax = plt.subplots(1,1, subplot_kw=dict(projection="mollweide"))
# ax.scatter(phis_hp, thetas_hp, c = 'red', s = 15, label  = 'Healpy')
for theta in thetas:
    for phi in phis:
        ax.scatter(phi, theta, c = 'k', s=20, marker = 'h')
plt.grid(True)
plt.legend()
plt.show()
