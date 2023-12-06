#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 18:38:23 2023

@author: konstantinos
"""

import sys
sys.path.append('/Users/paolamartire/tde_comparison')
import numpy as np
import matplotlib.pyplot as plt
import colorcet
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.figsize'] = [10, 8]
plt.rcParams['axes.facecolor']= 	'whitesmoke'
m = 4
Mbh = 10**m 
Rt =  Mbh**(1/3) # Msol = 1, Rsol = 1
apocenter = 2 * Rt * Mbh**(1/3)  # There is m_* hereeee

#%%
kind = 'early' # early mid late

def stacker(check, fixes):
    # Pathing BS
    if check == 'fid':
        path = 'data/den4-' + check
    if check == 'S60ComptonHires':
        path = 'data/den4-' + 'chr'
        
    for i in range(0, len(fixes)):
        # First
        if i == 0:
            den = np.loadtxt(path + '/denproj4-' + check + str(fixes[i]) + '.txt')
            continue
        den_new = np.loadtxt(path + '/denproj4-' + check + str(fixes[i]) + '.txt')

        # Stack
        den = np.add(den, den_new)
    # Mean
    inv_total_fixes = 1/len(fixes)
    den = np.multiply(den, inv_total_fixes)
    
    return den
def ypsilon_plot(den4, den4C, color = True):
    ypsilon = np.divide(den4, den4C)
    ypsilon = np.log10(ypsilon) # We want a log plot

    # Fix the fuckery
    ypsilon = np.nan_to_num(ypsilon, neginf=0)
    
    # Color re-normalization
    if color:
        ypsilon[ypsilon<0.1] = 0
        ypsilon[ypsilon>8] = 8
    
    # Transpose to look like we're used to
    ypsilon = ypsilon.T
    
    # Plot
    fig, ax = plt.subplots(1,1, tight_layout = True)
    
    # Images
    img = ax.pcolormesh( xs/apocenter, ys/apocenter , ypsilon, cmap='cet_coolwarm',
                        vmin = -2, vmax = 2)
    contours = ax.contour( xs/apocenter, ys/apocenter, ypsilon, levels = 6, 
                        colors = 'k',
                        vmin = -2, vmax = 2)
    ax.clabel(contours, inline=True, fontsize=15)
    fig.colorbar(img)

    plt.xlabel('X [x/R$_a$]', fontsize = 20)
    plt.ylabel('Y [y/R$_a$]', fontsize = 20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    
    if kind == 'early':
        txt = '0.78 - 0.87 t/t$_{FB}$'
    if kind == 'mid':
        txt = '0.95 - 1.05 t/t$_{FB}$'
    if kind == 'late':
        txt = '1.31 - 1.41 t/t$_{FB}$'
        
    plt.title(r' $\log (\rho / \tilde{\rho}) $ XY' + ' for ComptonHires : ' + txt ,
                  fontsize = 25)

pre = 'data/den4-fid'
xs = np.loadtxt(pre + '/xarray4-fid.txt') 
ys = np.loadtxt(pre + '/xarray4-fid.txt') 
if kind == 'late':
    fixes4 = np.arange(265, 276)
    fixes4CHR = np.arange(268, 278)
    den4 = stacker('fid', fixes4)
    den4CHR = stacker('S60ComptonHires', fixes4CHR)
    ypsilon_plot(den4, den4CHR)
if kind == 'mid':
    fixes4 = np.arange(226, 237)
    fixes4CHR = np.arange(230, 240)
    den4 = stacker('fid', fixes4)
    den4CHR = stacker('S60ComptonHires', fixes4CHR)
    ypsilon_plot(den4, den4CHR)
if kind == 'early':
    fixes4 = np.arange(208, 218)
    fixes4CHR = np.arange(210, 220)
    den4 = stacker('fid', fixes4)
    den4CHR = stacker('S60ComptonHires', fixes4CHR)
    ypsilon_plot(den4, den4CHR)

