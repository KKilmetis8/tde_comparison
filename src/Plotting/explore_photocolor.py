#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:11:11 2025

@author: konstantinos
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import colorcet
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import src.Utilities.prelude as c

pre = 'data/bluepaper/'

fixes4 = [179, 240, 300] 
fixes5 = [227, 288, 349] #
fixes6 = [315, 379, 444] # 420->444
zsweep = [104, 136, (152, 167), (168, 179), (180, 187), (188, 191)]#, 140]
angles = [9.6, 30, 42.4, 42.4, 55.3, 55.3, 68, 68, 81, 81]
colors = [c.darkb, c.cyan, c.prasinaki, c.yellow, c.kroki, c.reddish, c.c99]
labels = [ r'9.6°', '30°', '42.4°', '55.3°', '68°', '81°', '90°',]
reddata4 = np.genfromtxt(f'data/red/red_richex{4}.csv', delimiter = ',').T

mstar = 0.5
rstar = 0.47
Rt4 = rstar * (10**4/mstar)**(1/3)
amin4 = Rt4 * (10**4/mstar)**(1/3)
Rt5 = rstar * (10**5/mstar)**(1/3)
amin5 = Rt5 * (10**5/mstar)**(1/3)
Rt6 = rstar * (10**6/mstar)**(1/3)
amin6 = Rt6 * (10**6/mstar)**(1/3)
fig, ax = plt.subplots(3,3, figsize = (8,8), tight_layout = True
                      )
ax2s = []
for f4, f5, f6, i in zip(fixes4, fixes5, fixes6, range(3)):
    color4 = pd.read_csv(f'{pre}local_4photocolor{f4}.csv').iloc[-1][-1]
    photo4 = pd.read_csv(f'{pre}local_4photocolor{f4}.csv').iloc[-1][-2]
    color5 = pd.read_csv(f'{pre}local_5photocolor{f5}.csv').iloc[-1][-1]
    photo5 = pd.read_csv(f'{pre}local_5photocolor{f5}.csv').iloc[-1][-2]
    color6 = pd.read_csv(f'{pre}local_6photocolor{f6}.csv').iloc[-1][-1]
    photo6 = pd.read_csv(f'{pre}local_6photocolor{f6}.csv').iloc[-1][-2]
    color4 = list(map(float, color4.strip("[]").split()))
    photo4 = list(map(float, photo4.strip("[]").split()))
    color5 = list(map(float, color5.strip("[]").split()))
    photo5 = list(map(float, photo5.strip("[]").split()))
    color6 = list(map(float, color6.strip("[]").split()))
    photo6 = list(map(float, photo6.strip("[]").split()))
    
    photo_x4 = np.zeros(len(angles))
    photo_z4 = np.zeros(len(angles))
    color_x4 = np.zeros(len(angles))
    color_z4 = np.zeros(len(angles))
    photo_x5 = np.zeros(len(angles))
    photo_z5 = np.zeros(len(angles))
    color_x5 = np.zeros(len(angles))
    color_z5 = np.zeros(len(angles))
    photo_x6 = np.zeros(len(angles))
    photo_z6 = np.zeros(len(angles))
    color_x6 = np.zeros(len(angles))
    color_z6 = np.zeros(len(angles))
    for j in range(len(angles)):
        photo_x4[j] = photo4[j] * np.cos(angles[j] * np.pi / 180) / amin4
        photo_z4[j] = photo4[j] * np.sin(angles[j] * np.pi / 180) / amin4
        color_x4[j] = color4[j] * np.cos(angles[j] * np.pi / 180) / amin4
        color_z4[j] = color4[j] * np.sin(angles[j] * np.pi / 180) / amin4
        
        photo_x5[j] = photo5[j] * np.cos(angles[j] * np.pi / 180) / amin5
        photo_z5[j] = photo5[j] * np.sin(angles[j] * np.pi / 180) / amin5
        color_x5[j] = color5[j] * np.cos(angles[j] * np.pi / 180) / amin5
        color_z5[j] = color5[j] * np.sin(angles[j] * np.pi / 180) / amin5
        
        photo_x6[j] = photo6[j] * np.cos(angles[j] * np.pi / 180) / amin6
        photo_z6[j] = photo6[j] * np.sin(angles[j] * np.pi / 180) / amin6
        color_x6[j] = color6[j] * np.cos(angles[j] * np.pi / 180) / amin6
        color_z6[j] = color6[j] * np.sin(angles[j] * np.pi / 180) / amin6
    
    ax[i,0].plot(0,0, 's', c = 'k')
    ax[i,0].plot(photo_x4, photo_z4, '-o', c = c.AEK, lw = 2) 
    ax[i,0].plot(color_x4, color_z4, '-o', c = 'slateblue', lw = 1) 
    
    ax[i,1].plot(0,0, 's', c = 'k')
    ax[i,1].plot(photo_x5, photo_z5, '-o', c = c.AEK, lw = 2) 
    ax[i,1].plot(color_x5, color_z5, '-o', c = 'slateblue', lw = 1) 
    
    ax[i,2].plot(0,0, 's', c = 'k')
    ax[i,2].plot(photo_x6, photo_z6, '-o', c = c.AEK, lw = 2) 
    ax[i,2].plot(color_x6, color_z6, '-o', c = 'slateblue', lw = 1) 
    
    tidx4 = np.argmin(np.abs(f4 - reddata4[0]))
    time4 = reddata4[1][tidx4]
    ax[i,1].text(0.35, 0.05, f'{time4:.2f} $t_\mathrm{{FB}}$',
                 fontsize = 12, c = 'k', transform = ax[i,1].transAxes)  
    
    # x4max = 3
    # x5max = 5
    # x6max = 12
    # z4max = 2.5
    # z5max = 3.2
    # z6max = 16
    # ax[i,0].set_xlim(-0.1,x4max)
    # ax[i,1].set_xlim(-0.2,x5max)
    # ax[i,2].set_xlim(-1,x6max)
    # ax[i,0].set_ylim(-0.1,z4max)
    # ax[i,1].set_ylim(-0.2,z5max)
    # ax[i,2].set_ylim(-1,z6max)


ax[2,1].set_xlabel(r'X [x/$\alpha_\mathrm{min}$]', fontsize = 17)
ax[1,0].set_ylabel(r'Z [z/$\alpha_\mathrm{min}$]', fontsize = 17)
ax[0,0].set_title('$10^4 M_\odot$', fontsize = 17)
ax[0,1].set_title('$10^5 M_\odot$', fontsize = 17)
ax[0,2].set_title('$10^6 M_\odot$', fontsize = 17)
ax[0,1].set_xlim(-0.05, 1)
ax[0,1].set_ylim(-0.05, 0.4)

    
    
    
    