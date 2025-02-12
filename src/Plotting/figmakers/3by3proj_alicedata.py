#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:55:26 2024

@author: konstantinos
"""
# Vanilla
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import colorcet
import healpy as hp

# Choc
import src.Utilities.prelude as c

# Constants
rstar = 0.47
mstar = 0.5
Rt4 = rstar * (1e4/mstar)**(1/3)
amin4 = Rt4 * (1e4/mstar)**(1/3)
Rt5 = rstar * (1e5/mstar)**(1/3)
amin5 = Rt5 * (1e5/mstar)**(1/3)
Rt6 = rstar * (1e6/mstar)**(1/3)
amin6 = Rt6 * (1e6/mstar)**(1/3)
def find_sph_coord(r, theta, phi):
    x = r * np.sin(np.pi-theta) * np.cos(phi) #Elad has just theta
    y = r * np.sin(np.pi-theta) * np.sin(phi)
    z = r * np.cos(np.pi-theta)
    return [x,y,z]

def tuple_parse(strings):
    ''' parses "(1,2,3)" '''
    xs = np.zeros(len(strings))
    ys = np.zeros(len(strings))
    for i, string in enumerate(strings):
        values = string.strip("()").split(", ")
        tuple_values = tuple(np.float64(value.split("(")[-1].strip(")")) for value in values)
        xs[i] = tuple_values[0]
        ys[i] = tuple_values[1]
    return xs, ys

pre = 'data/denproj/paper/'
suf = 'beta1S60n1.5Compton'
when = 'test' # choices: early mid late test
plane = 'XY'
photo = False
if when == 'test':
    # 0.42, 0.82, 1.22
    fixes4 = [179, 240, 300] 
    fixes5 = [227, 288, 349] # 0.5, 1, 1.4
    fixes6 = [315, 379, 444] # 0.5, should be 376 and 414 but ok
    title_txt = 'Time: Trial t/t$_{FB}$'

size = 7
step = 1
fontsize = 17
fig, ax = plt.subplots(3,3, figsize = (1*1.4*size, size), sharex = True, sharey = True,
                       tight_layout = True)
for f4, f5, f6, i in zip(fixes4, fixes5, fixes6, range(3)):
    # Load projection data
    den4 = np.loadtxt(f'{pre}4normal{f4}.txt')[::step].T[::step].T
    x4 = np.loadtxt(f'{pre}4normalx.txt')[::step]
    y4 = np.loadtxt(f'{pre}4normaly.txt')[::step]
    
    den5 = np.loadtxt(f'{pre}5normal{f5}.txt')[::step].T[::step].T
    x5 = np.loadtxt(f'{pre}5normalx.txt')[::step]
    y5 = np.loadtxt(f'{pre}5normaly.txt')[::step]
    
    den6 = np.loadtxt(f'{pre}6normal{f6}.txt')[::step].T[::step].T
    x6 = np.loadtxt(f'{pre}6normalx.txt')[::step]
    y6 = np.loadtxt(f'{pre}6normaly.txt')[::step]
    
    reddata4 = np.genfromtxt(f'data/red/red_richex{4}.csv', delimiter = ',').T
    tidx4 = np.argmin(np.abs(f4 - reddata4[0]))
    time4 = reddata4[1][tidx4]
    reddata4 = np.genfromtxt(f'data/red/red_richex{5}.csv', delimiter = ',').T
    tidx5 = np.argmin(np.abs(f5 - reddata4[0]))
    time5 = reddata4[1][tidx5]
    reddata6 = np.genfromtxt(f'data/red/red_richex{6}.csv', delimiter = ',').T
    tidx6 = np.argmin(np.abs(f6 - reddata6[0]))
    time6 = reddata6[1][tidx6]
    
    # ax[i,0].text(0.78, 0.8, f'{time4:.2f} $t_\mathrm{{FB}}$', 
    #              fontsize = 12, c = 'white', transform = ax[i,0].transAxes)
    # ax[i,1].text(0.78, 0.8, f'{time4:.2f} $t_\mathrm{{FB}}$',
    #              fontsize = 12, c = 'white', transform = ax[i,1].transAxes)
    ax[i,2].text(0.1, 0.1, f'{time4:.2f} $t_\mathrm{{FB}}$',
                 fontsize = 12, c = 'white', transform = ax[i,2].transAxes)

    # Plot projection data
    dmin = 0.1
    dmax = 5
    img = ax[i,0].pcolormesh(x4/amin4, y4/amin4, den4.T, cmap = 'cet_fire',
                             vmin = dmin, vmax = dmax)
    ax[i,1].pcolormesh(x5/amin5, y5/amin5, den5.T, cmap = 'cet_fire',
                             vmin = dmin, vmax = dmax)
    ax[i,2].pcolormesh(x6/amin6, y6/amin6, den6.T, cmap = 'cet_fire',
                             vmin = dmin, vmax = dmax)
    
    # Plot Rt
    ax[i,0].add_patch(mp.Circle((0,0), Rt4/amin4, ls = '-', 
                                color = 'c', fill = False, lw = 1))
    ax[i,1].add_patch(mp.Circle((0,0), Rt5/amin5, ls = '-', 
                                color = 'c', fill = False, lw = 1))
    ax[i,2].add_patch(mp.Circle((0,0), Rt6/amin6, ls = '-',
                                color = 'c', fill = False, lw = 1))
    
    if i == 0:
        ax[i,0].set_title('10$^4$ M$_\odot$', fontsize = fontsize)
        ax[i,1].set_title('10$^5$ M$_\odot$', fontsize = fontsize)
        ax[i,2].set_title('10$^6$ M$_\odot$', fontsize = fontsize)

        # ax[i,0].set_title('0.42 $t_\mathrm{FB}$', fontsize = 17)
        # ax[i,1].set_title('0.82 $t_\mathrm{FB}$', fontsize = 17)
        # ax[i,2].set_title('1.22 $t_\mathrm{FB}$', fontsize = 17)

    # Plot photosphere

    if photo:
        import pandas as pd
        df4 = pd.read_csv('data/photosphere/richex2_photocolor4.csv', sep = ',',
                           comment = '#', header = None)
        
        photodata5 = np.genfromtxt('data/photosphere/photocolor5.csv', delimiter = ',')
        df6 = pd.read_csv('data/photosphere/richex_photocolor6.csv', sep = ',',
                          comment = '#', header = None)
        
        # Find snap in photodata
        idx4 = np.argmin(np.abs(f4 - df4.iloc[:,0]))
        idx5 = np.argmin(np.abs(f5 - photodata5.T[0]))
        idx6 = np.argmin(np.abs(f6 - df6.iloc[:,0]))
        
        # snap time photo color obs_num
        # Photosphere data is a 3-tuple for each observer
        # The equatorial observers are 88:104
        # good_obs = [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
        # photo_x4, photo_y4 = tuple_parse(df4.iloc[idx4][good_obs])
        # ax[i,0].plot(photo_x4 /amin4, photo_y4/amin4, '-o', 
        #              c = 'magenta', markersize = 3)
        # ax[i,1].plot(photo_x5/amin5, photo_y5/amin5, '-o', 
        #              c = 'magenta', markersize = 1)
        # ax[i,2].plot(photo_x6/amin6, photo_y6/amin6, '-o', 
        #              c = 'magenta', markersize = 1)
    
    # Set x-lims
    # 4
    xmin = -1.1
    xmax = 0.3
    ymin = -0.5
    ymax = 0.5
    ax[0,0].set_xlim(xmin, xmax)
    ax[1,0].set_xlim(xmin, xmax)
    ax[2,0].set_xlim(xmin, xmax)
    ax[0,0].set_ylim(ymin, ymax)
    ax[1,0].set_ylim(ymin, ymax)
    ax[2,0].set_ylim(ymin, ymax)
    
    # 5
    xmin = -1.1
    xmax = 0.3
    ymin = -0.5
    ymax = 0.5
    ax[0,1].set_xlim(xmin, xmax)
    ax[1,1].set_xlim(xmin, xmax)
    ax[2,1].set_xlim(xmin, xmax)
    ax[0,1].set_ylim(ymin, ymax)
    ax[1,1].set_ylim(ymin, ymax)
    ax[2,1].set_ylim(ymin, ymax)
    
    # 6
    xmin = -1.1
    xmax = 0.3
    ymin = -0.5
    ymax = 0.5
    ax[0,2].set_xlim(xmin, xmax)
    ax[1,2].set_xlim(xmin, xmax)
    ax[2,2].set_xlim(xmin, xmax)
    ax[0,2].set_ylim(ymin, ymax)
    ax[1,2].set_ylim(ymin, ymax)
    ax[2,2].set_ylim(ymin, ymax)

ax[2,1].set_xlabel(r'X $[\alpha_\mathrm{min}]$', fontsize = fontsize)
ax[1,0].set_ylabel(r'Y $[\alpha_\mathrm{min}]$', fontsize = fontsize)
cb = fig.colorbar(img, cax=fig.add_axes([1.02, 0.095, 0.04, 0.845]))
cb.set_label('$\log_{10} (\Sigma) $ [g/cm$^2$]', fontsize = fontsize)
cb.ax.tick_params(labelsize = fontsize-2)
for oneax in ax.flatten():
    oneax.tick_params(top=True, right=True, colors = 'whitesmoke',  
                      labelcolor = 'k', labelsize = fontsize - 2)

# plt.savefig('paperplots/denproj.eps')

    
    
    
