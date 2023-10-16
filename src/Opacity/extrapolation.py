#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: konstantinos, paola

Produce a new table already expanded, in order to interpolate here.
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.facecolor']= 	'whitesmoke'
import colorcet
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d

# All units are ln[cgs]
loadpath = 'src/Opacity/'
lnT = np.loadtxt(loadpath + 'T.txt')
lnrho = np.loadtxt(loadpath + 'rho.txt')
lnk_ross = np.loadtxt(loadpath + 'ross.txt')
lnk_planck = np.loadtxt(loadpath + 'planck.txt')
lnk_scatter = np.loadtxt(loadpath + 'scatter.txt')

# Minimum we need is 3.99e-22, Elad's lnrho stops at 1e-10
kind = 'rosseland'
save = False

rho_min = np.log(3.99e-22)
rho_max = np.log(8e-11)
expanding_rho = np.arange(rho_min,rho_max, 0.2)
table_expansion = np.zeros( (len(lnT), len(expanding_rho) ))

for i, T in enumerate(lnT):
    if kind == 'rosseland':
        opacity_col = lnk_ross[i] 
    elif kind == 'planck':
        opacity_col = lnk_planck[i]
    elif kind == 'scatter':
        opacity_col = lnk_scatter[i]
        
        
    extra = interp1d(lnrho, opacity_col, kind = 'linear',
                        fill_value = 'extrapolate')
    for j, rho in enumerate(expanding_rho):           
        opi = extra(rho)
        table_expansion[i,j] = opi

# Combine
new_rho = np.concatenate((expanding_rho, lnrho))
new_table = np.concatenate( (table_expansion, lnk_ross), axis = 1)

if save:
    if kind == 'rosseland':
        np.savetxt(loadpath + 'ross_expansion.txt', new_table)
    elif kind == 'planck':
        np.savetxt(loadpath + 'planck_expansion.txt', new_table)
    elif kind == 'scatter':
        np.savetxt(loadpath + 'scatter_expansion.txt', new_table)
            
    np.savetxt(loadpath + 'big_lnrho.txt', new_rho)

# Plotting
plot = True
if plot:
    # Norm
    cmin = -45
    cmax = 21
    
    if kind == 'rosseland':
        k = lnk_ross 
    elif kind == 'planck':
        k = lnk_planck 
    elif kind == 'scatter':
        k = lnk_scatter
        
    # Elad's Table
    # fig  = plt.figure( figsize = (6,4))
    # img = plt.pcolormesh(lnrho, lnT, k, 
    #                       cmap = 'cet_fire', vmin = cmin, vmax = cmax)

    # plt.xlabel(r'$\ln ( \rho )$ $[g/cm^3]$')
    # plt.ylabel('$\ln(T)$ $[K]$')
    # plt.title('Rosseland Mean Opacity | Elads Table')
        
    # cax = fig.add_axes([0.93, 0.125, 0.04, 0.76])
    # cbar = fig.colorbar(img, cax=cax)
    # cbar.set_label('$\ln(\kappa)$ $[cm^-1]$', rotation=270, labelpad = 15)
    
    # Extrapolated Table
    fig = plt.figure( figsize = (8,4) )
    img = plt.pcolormesh(new_rho, lnT, new_table, 
                          cmap = 'cet_fire', vmin = cmin, vmax = cmax)
    plt.xlabel(r'$\ln( \rho )$ $[g/cm^3]$')
    plt.ylabel('$\ln(T)$ $[K]$')
    plt.title('Rosseland Mean Opacity | Extrapolated Table')
    plt.axvline( (expanding_rho[-1] + lnrho[0]) /2 , 
                color = 'b', linestyle = 'dashed')
    
    cax = fig.add_axes([0.92, 0.125, 0.03, 0.76])
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label('$\ln(\kappa)$ $[cm^{-1}]$', rotation=270, labelpad = 15)


   