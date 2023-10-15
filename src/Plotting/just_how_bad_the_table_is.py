#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 19:22:56 2023

@author: konstantinos
"""
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.facecolor']= 	'whitesmoke'
import colorcet
from sklearn.neighbors import KernelDensity as  kde

fig, ax = plt.subplots( figsize = (8,4) )
img = plt.pcolormesh(new_rho, lnT, new_table, 
                      cmap = 'cet_fire', vmin = cmin, vmax = cmax)
ax.set_xlabel(r'$\ln( \rho )$ $[g/cm^3]$')
ax.set_ylabel('$\ln(T)$ $[K]$')
ax.set_title('Rosseland Mean Opacity | Extrapolated Table')
ax.axvline( (expanding_rho[-1] + lnrho[0]) /2 , 
            color = 'b', linestyle = 'dashed')

cax = fig.add_axes([0.92, 0.125, 0.03, 0.76])
cbar = fig.colorbar(img, cax=cax)
cbar.set_label('$\ln(\kappa)$ $[cm^{-1}]$', rotation=270, labelpad = 15)

ax.plot(np.log(np.array(rays_den).ravel()), np.log(np.array(rays_T).ravel()), 
        'x', c='g', markersize = 1)