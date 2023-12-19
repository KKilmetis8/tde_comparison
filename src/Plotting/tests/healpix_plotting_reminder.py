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
hp.mollview(m, title="Mollview image RING")
hp.graticule()
plt.savefig('Figs/mollview.png')

# Healpix and my plots
thetas_hp = np.zeros(192)
phis_hp = np.zeros(192)

for i in range(0,192):
   thetas_hp[i], phis_hp[i] = hp.pix2ang(NSIDE, i)
   # thetas_hp[i] -= np.pi/2
   # phis_hp[i] -= np.pi
   #print('theta:', thetas_hp[i], ',')
   #print('phi:', phis_hp[i] , '\n')

for i in range(0,len(thetas_hp)-1):
    print('theta', thetas_hp[i+1]-thetas_hp[i])
    print('phi', phis_hp[i+1]-phis_hp[i])

thetas_toplot = thetas_hp - np.pi/2
phis_toplot = phis_hp - np.pi
fig, ax = plt.subplots(1,1, subplot_kw=dict(projection="mollweide"))
ax.scatter(phis_toplot, thetas_toplot, c = np.arange(192), s=20, marker = 'h')
plt.grid(True)
plt.legend()
#plt.show()
