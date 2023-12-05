#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 11:28:20 2023

@author: konstantinos

Ypsilon test
"""

import numpy as np

# Custom Imports
from src.Calculators.casters import THE_CASTER
from src.Calculators.casters_split import THE_CASTER_SPLIT_2
from src.Calculators.reducers import THE_REDUCER_2, THE_FILTERER, THE_PIXELATOR
from src.Extractors.time_extractor import days_since_distruption

# Pretty plots
import matplotlib.pyplot as plt
import colorcet as cc # cooler colormaps
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [16 , 16]

check1 = 'base'
check2 = 'hr4'
sims = [check1 + '-', check2 + '-']
fixes = [217]
    
def gridder(check, sim, fix, project = False):
    m = 4
    Mbh = 10**m
    Rt =  Mbh**(1/3) # Msol = 1, Rsol = 1
    apocenter = 2 * Rt * Mbh**(1/3)
    fix = str(fix)
    name = 'convergence/' + fix + '/' + sim + fix

    days = str(np.round(days_since_distruption(name + '.h5'),3))
    # CM Position Data
    X = np.load(name +'CMx.npy')
    Y = np.load(name + 'CMy.npy')
    Z = np.load(name + 'CMz.npy')
    # Import Density
    Den = np.load(name + 'Den.npy')
    Vol = np.load(name + 'Volume.npy')
    # Mass = np.load(name + 'Mass.npy')
        
    # Need to convert Msol/Rsol^2 to g/cm
    Msol_to_g = 1.989e33
    Rsol_to_cm = 6.957e10
    converter = Msol_to_g / Rsol_to_cm**2
    Den *=  converter
    
    # Specify new grid:
    x_start = -400 # -apocenter - 4 *2*Rt
    x_stop = 100 # 10 * 2*Rt
    x_num = 200 # np.abs(x_start - x_stop)
    xs = np.linspace(x_start, x_stop, num = x_num )
    # y +- 150, z +- 50
    y_start = -200 # -apocenter
    y_stop = 200 # apocenter
    y_num = 100 # np.abs(y_start - y_stop)
    # x_start = -apocenter - 4 *2*Rt
    # x_stop = 10 * 2*Rt
    # x_num = 200 # np.abs(x_start - x_stop)
    # xs = np.linspace(x_start, x_stop, num = x_num )
    # # y +- 150, z +- 50
    # y_start = -apocenter
    # y_stop = apocenter
    # y_num = 100 # np.abs(y_start - y_stop)
    
    ys = np.linspace(y_start, y_stop, num = y_num)    
    z_start = -100
    z_stop = 100
    z_num = 10 # np.abs(y_start - y_stop)
    zs = np.linspace(z_start, z_stop, num = z_num)    
    # EVOKE
    pixels = THE_PIXELATOR(xs, ys, zs)
    nX, nY, nZ, nDen = THE_FILTERER(X, Y, Z, Den, Vol)
    shape = (x_num, y_num, z_num)
    den_cast = THE_REDUCER_2(pixels, shape, 
                             nX, nY, nZ, 
                             nDen)
    
    # Remove bullshit and fix things
    den_cast = np.nan_to_num(den_cast) # need the T here
    
    if project:
        den_project = np.zeros( (shape[0], shape[1]) )
        for i in range(len(zs)):
            den_project = np.add(den_project, den_cast[...,i])
    
        return den_project, xs, ys, days
    
    return den_cast, xs, ys, days

def plot_prepare(den_cast, color = True):
    # We want a log plot
    den_cast = np.log10(den_cast) 
    
    # Fix the fuckery
    den_cast = np.nan_to_num(den_cast, neginf=0)
    
    # Color re-normalization
    if color:
        den_cast[den_cast<0.1] = 0
        den_cast[den_cast>8] = 8
    
    # Transpose to look like we're used to
    den_cast = den_cast.T
    return den_cast

def ypsilon_maker(den_baseline, den_check):
    ypsilon = np.divide(den_baseline, den_check)
    ypsilon = plot_prepare(ypsilon, color = False)
    return ypsilon
    
for fix in fixes:
    den_baseline, xs, ys, day1 = gridder(check1, sims[0], fix, project = True)
    den_check, _, _, day2 = gridder(check2, sims[1], fix, project = True)
    ypsilon = ypsilon_maker(den_baseline, den_check)
    den_baseline = plot_prepare(den_baseline)
    den_check = plot_prepare(den_check)
#%% Plotting
fig, ax = plt.subplots(3,1, tight_layout = True)

# Images
img = ax[0].pcolormesh(xs, ys, ypsilon, cmap='cet_coolwarm',
                        vmin = -2, vmax = 2)
fig.colorbar(img)

img1 = ax[1].pcolormesh(xs, ys, den_baseline, cmap='cet_fire',
                        vmin = 0, vmax = 8)
fig.colorbar(img1)

img2 = ax[2].pcolormesh(xs, ys, den_check, cmap='cet_fire',
                        vmin = 0, vmax = 8)
fig.colorbar(img2)

fig.suptitle(r'$\upsilon = \log (\rho / \tilde{\rho}) $ XY' + ' for HiRez4',
              fontsize = 55)
ax[1].text(0.1, 0.1, check1 + ': ' + day1,
            fontsize = 50,
            color='white', fontweight = 'bold', 
            transform=ax[1].transAxes)
ax[2].text(0.1, 0.1, check2 + ': ' + day2,
            fontsize = 50,
            color='white', fontweight = 'bold', 
            transform=ax[2].transAxes)




