#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 20:41:34 2023

@author: konstantinos
"""

import numpy as np
# Pretty plots
import matplotlib.pyplot as plt
import colorcet # cooler colormaps
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [16 , 16]

# Specify new grid:
x_start = -400 # -apocenter - 4 *2*Rt
x_stop = 100 # 10 * 2*Rt
x_num = 200 # np.abs(x_start - x_stop)
xs = np.linspace(x_start, x_stop, num = x_num )
# y +- 150, z +- 50
y_start = -200 # -apocenter
y_stop = 200 # apocenter
y_num = 100 # np.abs(y_start - y_stop)
ys = np.linspace(y_start, y_stop, num = y_num)    

# Data load
temperature = False
radiation = True
if temperature:
    prefix = 'T-'
elif radiation:
    prefix = 'rad-'
else:
    prefix = ''

# Data load
check1 = 'new'
check2 = 'compton'

# HR4
# days = '3.027-3.506' 
# days = '2.373-2.879'
# days = '3.58-3.912'

# Compton
# days = '1.746-2.41'
days = '2.447-3.137'

# S10
#days = '1.494-1.968'
#days = '2.041-2.632'

ypsilon = np.load('products/convergance/' + prefix +'ypsilon-' + check1 + '-' + check2 + '-' + days + '.npy')
den_baseline = np.load('products/convergance/' + prefix +'proj-' + check1 + '-' + days + '.npy')
den_check = np.load('products/convergance/' + prefix +'proj-' + check2 + '-' + days + '.npy')

fig, ax = plt.subplots(3,1, tight_layout = True)

# Images
img = ax[0].pcolormesh( xs, ys , ypsilon, cmap='cet_coolwarm',
                        vmin = -1, vmax = 1)
contours = ax[0].contour( xs, ys , ypsilon, levels = 6, 
                        colors = 'k',
                        vmin = -1, vmax = 1)
ax[0].clabel(contours, inline=True, fontsize=20)
fig.colorbar(img)

img1 = ax[1].pcolormesh(xs, ys, den_baseline, cmap='cet_fire',
                        vmin = 0, vmax = 10)
fig.colorbar(img1)

img2 = ax[2].pcolormesh(xs, ys, den_check, cmap='cet_fire',
                        vmin = 0, vmax = 10)
fig.colorbar(img2)

if prefix == "":
    fig.suptitle(r'$\upsilon = \log (\rho / \tilde{\rho}) $ XY' + ' for S10',
                  fontsize = 55)
elif prefix =="T-":
    fig.suptitle(r'$\upsilon = \log (T / \tilde{T}) $ XY' + ' for Compton',
                  fontsize = 55)
elif prefix =="rad-":
    fig.suptitle(r'$\upsilon = \log (E_{rad} \cdot \rho / \tilde{E_{rad}} \tilde{\rho}) $ XY' + ' for Compton',
                  fontsize = 55)
ax[1].text(0.1, 0.1, check1 + ': ' + days,
            fontsize = 40,
            color='white', fontweight = 'bold', 
            transform=ax[1].transAxes)
ax[2].text(0.1, 0.1, check2 + ': ' + days,
            fontsize = 50,
            color='white', fontweight = 'bold', 
            transform=ax[2].transAxes)