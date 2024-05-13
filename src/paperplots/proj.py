#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 15:14:37 2024

@author: konstantinos
"""
# Vanilla
import os
import numpy as np
import matplotlib.pyplot as plt
import colorcet
from scipy.spatial import KDTree
from tqdm import tqdm

# Choc
import src.Utilities.prelude as c
from src.Calculators.casters import THE_CASTER

#%%
def maker(m, pixel_num, fix, plane, thing, how):
    Mbh = 10**m
    Rt =  Mbh**(1/3) # Msol = 1, Rsol = 1
    # t_fall = 40 * (Mbh/1e6)**(0.5) # days EMR+20 p13
    apocenter = 2 * Rt * Mbh**(1/3)
    pre = str(m) + '/' + fix
    
    # days = np.round( days_since_distruption(pre+'/snap_'+fix+'.h5') / t_fall, 1)
    Mass = np.load(pre + '/Mass_' + fix + '.npy')

    Den = np.load(pre + '/Den_' + fix + '.npy')
    # Need to convert Msol/Rsol^2 to g/cm
    Msol_to_g = 1.989e33
    Rsol_to_cm = 6.957e10
    converter = Msol_to_g / Rsol_to_cm**2
    Den *=  converter


    # CM Position Data
    X = np.load(pre + '/CMx_' + fix + '.npy')
    Y = np.load(pre + '/CMy_' + fix + '.npy')
    x_start = -apocenter
    x_stop = 0.2 * apocenter
    x_num = pixel_num # np.abs(x_start - x_stop)
    xs = np.linspace(x_start, x_stop, num = x_num )
    y_start = -0.12 * apocenter 
    y_stop = 0.12 * apocenter
    y_num = pixel_num # np.abs(y_start - y_stop)
    ys = np.linspace(y_start, y_stop, num = y_num)
    
    if how == 'caster':
        den_cast = THE_CASTER(xs, X, ys, Y, Den) # EVOKE

    # Make a tree
    if how == 'tree':
        Z = np.load(pre + '/CMz_' + fix + '.npy')

        sim_value = [X, Y, Z] 
        sim_value = np.transpose(sim_value) #array of dim (number_points, 3)
        sim_tree = KDTree(sim_value) 
        
        z_start = -2 * Rt
        z_stop = 2 * Rt
        z_num = 100
        z_radii = np.linspace(z_start, z_stop, z_num) #simulator units

        gridded_indexes =  np.zeros(( len(xs), len(ys), len(z_radii) ))
        den_cast =  np.zeros(( len(xs), len(ys), len(z_radii) ))
        gridded_mass =  np.zeros(( len(xs), len(ys), len(z_radii) ))
        for i in tqdm(range(len(xs))):
            for j in range(len(ys)):
                for k in range(len(z_radii)):
                    queried_value = [xs[i], ys[j], z_radii[k]]
                    _, idx = sim_tree.query(queried_value)
                                        
                    # Store
                    gridded_indexes[i, j, k] = idx
                    den_cast[i, j, k] = Den[idx]
        #             gridded_mass[i,j, k] = Mass[idx]
        # den_cast = np.divide(den_cast, gridded_mass)
        den_cast = den_cast[:,:,len(z_radii)//2]

    # Remove bullshit and fix things
    den_cast = np.nan_to_num(den_cast.T)
    den_cast = np.log10(den_cast) # we want a log plot
    den_cast = np.nan_to_num(den_cast, neginf=0) # fix the fuckery
        
    # Color re-normalization
    #if thing == 'Den':
    if how == 'caster':
        den_cast[den_cast<1] = 0
        den_cast[den_cast>8] = 8
    if how == 'tree':
        den_cast[den_cast<0.2] = 0
        den_cast[den_cast>5] = 5

    return xs/apocenter, ys/apocenter, den_cast# , days

#%%
plane = 'XY'
thing = 'Den' # Den or T
when = 'trial' # early mid late last
if when == 'trial':
    fixes4 = ['200'] # 0.65
    fixes5 = ['259'] # 0.55
    fixes6 = ['683'] # 0.5
    title_txt = 'Time: Trial t/t$_{FB}$'

# for i in range(len(fixes4)):

#     # Plotting
#     fig, ax = plt.subplots(3, 3, clear = True, tight_layout = True)
    
#     # Image making
#     x4, y4, d4 = maker(4, 500, fixes4[i], plane, thing)
#     ax[0,0].pcolormesh(x4, y4, d4, cmap='cet_fire', vmin = 0, vmax = 8)
#     del x4, y4, d4
#     x5, y5, d5 = maker(5, 500, fixes5[i], plane, thing)
#     img2 = ax[0,1].pcolormesh(x5, y5, d5, cmap='cet_fire', vmin = 0, vmax = 8)
#     del x5, y5, d5
#     x6, y6, d6 = maker(6, 500, fixes6[i], plane, thing)
#     ax[0,2].pcolormesh(x6, y6, d6, cmap='cet_fire', vmin = 0, vmax = 8)
#     del x6, y6, d6
#%%
i = 0
how = 'tree'
if how == 'tree':
    res = 200
else:
    res = 1000
    
x4, y4, d4 = maker(4, res, fixes4[i], plane, thing, how)
x5, y5, d5 = maker(5, res, fixes5[i], plane, thing, how)
x6, y6, d6 = maker(6, res, fixes6[i], plane, thing, how)
#%%
# Plotting
fig, ax = plt.subplots(3, 3, clear = True, tight_layout = True, 
                       figsize = (13,13))
if how == 'tree':
    vmax = 5
else:
    vmax = 8
# Image making
ax[0,0].pcolormesh(x4, y4, d4, cmap='cet_fire', vmin = 0, vmax = vmax)
img2 = ax[0,1].pcolormesh(x5, y5, d5, cmap='cet_fire', vmin = 0, vmax = vmax)
ax[0,2].pcolormesh(x6, y6, d6, cmap='cet_fire', vmin = 0, vmax = vmax)

cax = fig.add_axes([1.03, 0.045, 0.035, 0.94])
cb = fig.colorbar(img2, cax=cax)  
cb.ax.tick_params(labelsize=25, pad = 5)
cb.set_label(r'Density $\log_{10}(\rho)$ [g/cm$^2$]', fontsize = 25, labelpad = 15)
# Axis labels
# fig.text(0.5, -0.03, plane[0] + r' [x/R$_a$]', ha='center', fontsize = 25)
# fig.text(-0.02, 0.5, plane[1] + r' [y/R$_a$]', va='center', rotation='vertical', fontsize = 25)
ax[2,1].set_xlabel(plane[0] + r' [x/R$_a$]', ha='center', fontsize = 25 )
ax[1,0].set_ylabel(plane[1] + r' [y/R$_a$]', ha='center', fontsize = 25 )

#plt.savefig('xyproj.png')#, format = 'pdf', dpi = 400)
    
from src.Utilities.finished import finished
finished()