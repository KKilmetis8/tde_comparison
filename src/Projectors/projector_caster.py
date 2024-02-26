#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 12:45:18 2023

@author: konstantinos
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [8 , 4]
import numba
import matplotlib.colors as colors
import colorcet as cc # cooler colormaps
from src.Calculators.casters import THE_CASTER
# from scipy import ndimage # rotates image

# Choose black hole
m = 4
Mbh = 10**m
Rt =  Mbh**(1/3) # Msol = 1, Rsol = 1
apocenter = 2 * Rt * Mbh**(1/3)
    
# Choose snapshot
if m == 4:
    fixes = ['394'] # ['177', '232', '263']
if m == 6:
    fixes = ['844']

choice = 'XY'
quant = 'T' # Den or T

def maker(fix, choice):
    pre = str(m) + '/'
    Mass = np.load(pre+fix + '/Mass_' + fix + '.npy')

    if quant == 'Den':
        Den = np.load(pre+fix + '/Den_' + fix + '.npy')
        # Need to convert Msol/Rsol^2 to g/cm
        Msol_to_g = 1.989e33
        Rsol_to_cm = 6.957e10
        converter = Msol_to_g / Rsol_to_cm**2
        Den *=  converter
    
    if quant == 'T':
        Den = np.load(pre+fix + '/T_' + fix + '.npy')

    if choice == 'XY':
        # CM Position Data
        X = np.load(pre + fix + '/CMx_' + fix + '.npy')
        Y = np.load(pre + fix + '/CMy_' + fix + '.npy')
        if m == 6:
            x_start = -200
            x_stop = 200
            x_num = 100 # np.abs(x_start - x_stop)
            xs = np.linspace(x_start, x_stop, num = x_num )
            y_start = -100
            y_stop = 100
            y_num = 100 # np.abs(y_start - y_stop)
            ys = np.linspace(y_start, y_stop, num = y_num)
        
        if m==4:
            x_start = -apocenter
            x_stop = apocenter
            x_num = 1000 # np.abs(x_start - x_stop)
            xs = np.linspace(x_start, x_stop, num = x_num )
            y_start = -apocenter 
            y_stop = apocenter
            y_num = 1000 # np.abs(y_start - y_stop)
            ys = np.linspace(y_start, y_stop, num = y_num)
            
    if choice == 'XZ':
        X = np.load(fix + '/CMx_' + fix + '.npy')
        Y = np.load(fix + '/CMz_' + fix + '.npy')
        if m == 4:
            x_start = -apocenter
            x_stop = 1000
            x_num = 300 # np.abs(x_start - x_stop)
            xs = np.linspace(x_start, x_stop, num = x_num )
            y_start = -300
            y_stop = 300
            y_num = 300 # np.abs(y_start - y_stop)
            ys = np.linspace(y_start, y_stop, num = y_num)
        
    if choice == 'YZ':
        if m == 4:
            X = np.load(fix + '/CMy_' + fix + '.npy')
            Y = np.load(fix + '/CMz_' + fix + '.npy')
            x_start = -300
            x_stop = 200
            x_num = 300 # np.abs(x_start - x_stop)
            xs = np.linspace(x_start, x_stop, num = x_num )
            y_start = -150
            y_stop = 150
            y_num = 300 # np.abs(y_start - y_stop)
            ys = np.linspace(y_start, y_stop, num = y_num)
        
    # EVOKE
    den_cast = THE_CASTER(xs, X, ys, Y, Den)# , weights = Mass)
    
    # Remove bullshit and fix things
    den_cast = np.nan_to_num(den_cast.T)
    den_cast = np.log10(den_cast) # we want a log plot
    den_cast = np.nan_to_num(den_cast, neginf=0) # fix the fuckery
    
    # Color re-normalization
    if quant == 'Den':
        den_cast[den_cast<1] = 0
        den_cast[den_cast>5] = 5
    if quant == 'T':
        den_cast[den_cast<1] = 0
        den_cast[den_cast>10] = 10
    return den_cast, xs, ys
        
        
for fix in fixes:    
    den_cast, xs, ys = maker(fix, choice)
    #%%
    fig, ax = plt.subplots()
    img = ax.pcolormesh(xs, ys, den_cast, cmap='cet_fire', vmin = 0, vmax = 10) # vmin = 15, vmax = 25)
    fig.colorbar(img)
    
    # ax.set_ylim(-50, 50)
    # ax.set_xlim(-1500, 1000)
    if quant == 'Den':
        ax.set_title( choice + r' Density $\log_{10}( \rho )$ [g/cm$^2$] projection',
                     fontsize = 20)
    if quant == 'T':
        ax.set_title( choice + r' Temprature $\log_{10}(T)$ [K] projection',
                     fontsize = 20)
    
    ax.set_xlabel(choice[0]+'-coordinate $[R_{\odot}$]')
    ax.set_ylabel(choice[1]+'-coordinate $[R_{\odot}$]')
    
    # txt_x = xs[0] + 50
    # txt_y = ys[0] + 50
    
    # ax.text(txt_x, txt_y, 'Days after disruption: ' + days,
    #         color='white', 
    #         fontweight = 'bold', 
    #         fontname = 'Consolas',
    #         fontsize = 18)
    
    # from src.Utilities.finished import finished
    # finished()
