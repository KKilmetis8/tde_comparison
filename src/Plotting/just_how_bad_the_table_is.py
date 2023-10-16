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

fig, ax = plt.subplots( figsize = (8,4) )
img = plt.pcolormesh(np.log10(np.exp(new_rho)), np.log10(np.exp(lnT)), new_table, 
                      cmap = 'cet_fire', vmin = cmin, vmax = cmax)
ax.set_xlabel(r'$\log( \rho )$ $[g/cm^3]$')
ax.set_ylabel('$\log(T)$ $[K]$')
ax.set_title('Rosseland Mean Opacity | Extrapolated Table')
ax.axvline( np.log10(np.exp(lnrho[0])) , 
            color = 'b', linestyle = 'dashed')

cax = fig.add_axes([0.92, 0.125, 0.03, 0.76])
cbar = fig.colorbar(img, cax=cax)
cbar.set_label('$\ln(\kappa)$ $[cm^{-1}]$', rotation=270, labelpad = 15)

ax.plot(np.log10(np.array(rays_den).ravel()), np.log10(np.array(rays_T).ravel()), 
        'x', c='b', markersize = 1)

#%%
fig, ax = plt.subplots( figsize = (8,4) )
ax.axvline( np.log10(np.exp(-23)) , 
            color = 'b', linestyle = 'dashed')
ax.axvline( np.log10(np.exp(-0.18)) , 
            color = 'b', linestyle = 'dashed')
ax.axhline( np.log10(np.exp(8.666)) , 
            color = 'r', linestyle = 'dashed')
ax.axhline( np.log10(np.exp(17.876)) , 
            color = 'r', linestyle = 'dashed')

# ax.plot(np.log10(Den_triple_casted.ravel() ), np.log10(T_triple_casted.ravel()), 
#         'x', c='k', markersize = 1)
# ax.plot([],[], 'x', c = 'k', label = 'Triple' )
ax.plot(np.log10(np.array(rays_den).ravel()), np.log10(np.array(rays_T).ravel()), 
        'x', c='g', markersize = 1, alpha = 0.1)
ax.plot([],[], 'x', c = 'g', label = 'Spherical' )

ax.legend( loc = 'upper right')
ax.grid()
ax.set_xlabel(r'$\log( \rho )$ $[g/cm^3]$')
ax.set_ylabel('$\log(T)$ $[K]$')