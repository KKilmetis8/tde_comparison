#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 18:01:54 2023

@author: konstantinos
"""
import numpy as np
import os


from src.Calculators.casters import THE_CASTER
from src.Extractors.time_extractor import days_since_distruption
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [9 , 8]
import colorcet
#%%
def maker(m, pixel_num, fix, plane, thing):
    Mbh = 10**m
    Rt =  Mbh**(1/3) # Msol = 1, Rsol = 1
    t_fall = 40 * (Mbh/1e6)**(0.5) # days EMR+20 p13
    apocenter = 2 * Rt * Mbh**(1/3)
    pre = str(m) + '/' + fix
    # days = np.round( days_since_distruption(pre+'/snap_'+fix+'.h5') / t_fall, 1)
    Mass = np.load(pre + '/Mass_' + fix + '.npy')

    if thing == 'Den':
        Den = np.load(pre + '/Den_' + fix + '.npy')
        # Need to convert Msol/Rsol^2 to g/cm
        Msol_to_g = 1.989e33
        Rsol_to_cm = 6.957e10
        converter = Msol_to_g / Rsol_to_cm**2
        Den *=  converter
    
    if thing == 'T':
        Den = np.load(pre + '/T_' + fix + '.npy')

    if plane == 'XY':
        # CM Position Data
        X = np.load(pre + '/CMx_' + fix + '.npy')
        Y = np.load(pre + '/CMy_' + fix + '.npy')
        if m == 6:
            x_start = -apocenter
            x_stop = 0.2 * apocenter
            x_num = pixel_num # np.abs(x_start - x_stop)
            xs = np.linspace(x_start, x_stop, num = x_num )
            y_start = -0.1 * apocenter 
            y_stop = 0.15 * apocenter
            y_num = pixel_num # np.abs(y_start - y_stop)
            ys = np.linspace(y_start, y_stop, num = y_num)
        
        if m==4:
            x_start = -apocenter
            x_stop = 0.2 * apocenter
            x_num = pixel_num # np.abs(x_start - x_stop)
            xs = np.linspace(x_start, x_stop, num = x_num )
            y_start = -500 
            y_stop = 500
            y_num = pixel_num # np.abs(y_start - y_stop)
            ys = np.linspace(y_start, y_stop, num = y_num)
            
    if plane == 'XZ':
        X = np.load(pre + '/CMx_' + fix + '.npy')
        Y = np.load(pre + '/CMz_' + fix + '.npy')
        if m == 4:
            x_start = -apocenter
            x_stop = 250
            x_num = pixel_num # np.abs(x_start - x_stop)
            xs = np.linspace(x_start, x_stop, num = x_num )
            y_start = -300
            y_stop = 300
            y_num = pixel_num # np.abs(y_start - y_stop)
            ys = np.linspace(y_start, y_stop, num = y_num)
        if m == 6:
            x_start = -apocenter
            x_stop = 0.1 * apocenter
            x_num = pixel_num # np.abs(x_start - x_stop)
            xs = np.linspace(x_start, x_stop, num = x_num )
            y_start = -1000
            y_stop = 1000
            y_num = pixel_num # np.abs(y_start - y_stop)
            ys = np.linspace(y_start, y_stop, num = y_num)
        
    if plane == 'YZ':
        X = np.load(pre + '/CMy_' + fix + '.npy')
        Y = np.load(pre + '/CMz_' + fix + '.npy')
        if m == 4:
            x_start = -300
            x_stop = 200
            x_num = pixel_num # np.abs(x_start - x_stop)
            xs = np.linspace(x_start, x_stop, num = x_num )
            y_start = -150 # Msol = 1, Rsol = 1
            y_stop = 150
            y_num = pixel_num # np.abs(y_start - y_stop)
            ys = np.linspace(y_start, y_stop, num = y_num)
        if m == 6:
            x_start = -2000
            x_stop = 2000
            x_num = pixel_num # np.abs(x_start - x_stop)
            xs = np.linspace(x_start, x_stop, num = x_num )
            y_start = -1000
            y_stop = 1000
            y_num = pixel_num # np.abs(y_start - y_stop)
            ys = np.linspace(y_start, y_stop, num = y_num)
    
    # EVOKE
    if m == 6 and thing != 'T':
        den_cast = THE_CASTER(xs, X, ys, Y, Den) #, weights = Mass)
    if m ==  4 or thing == 'T':
        den_cast = THE_CASTER(xs, X, ys, Y, Den , weights = Mass)
    
    # Remove bullshit and fix things
    den_cast = np.nan_to_num(den_cast.T)
    den_cast = np.log10(den_cast) # we want a log plot
    den_cast = np.nan_to_num(den_cast, neginf=0) # fix the fuckery
    
    # Color re-normalization
    if thing == 'Den':
        den_cast[den_cast<1] = 0
        den_cast[den_cast>8] = 8
    if thing == 'T':
        den_cast[den_cast<1] = 0
        den_cast[den_cast>8] = 8
    
    return xs/apocenter, ys/apocenter, den_cast# , days

#%%
plane = 'XY'
thing = 'Den' # Den or T
when = 'last' # early mid late last
if when == 'early':
    fixes4 = ['172']
    fixes6 = ['683']
    title_txt = 'Time: 0.5 t/t$_{FB}$'
if when == 'mid':
    fixes4 = ['232']
    fixes6 = ['844']
    title_txt = 'Time: 1 t/t$_{FB}$'
if when == 'late':
    fixes4 = ['282']
    fixes6 = ['980']
    title_txt = 'Time: 1.5 t/t$_{FB}$'
if when == 'last':
    fixes4 = ['322']
    fixes6 = ['1008']
    title_txt = 'Time: Last available t/t$_{FB}$'



for i in range(len(fixes4)):
    x4, y4, d4 = maker(4, 1000, fixes4[i], plane, thing)
    # x6, y6, d6 = maker(6, 1_000, fixes6[i], plane, thing)
    # Plotting
    fig, ax = plt.subplots(1, 1, num = 1, clear = True, tight_layout = True)
    # Image making
    # img1 = ax[0].pcolormesh(x6, y6, d6, cmap='cet_fire', vmin = 0, vmax = 6)
    # plt.colorbar(img1)
    img2 = ax.pcolormesh(x4, y4, d4, cmap='cet_fire', vmin = 0, vmax = 6)
    
    cax = fig.add_axes([1, 0.045, 0.035, 0.88])
    fig.colorbar(img2, cax=cax)
    
    # Days text
    # txt_x = 0.86
    # txt_y = 0.9 
    # txt2 = ax[0].text(txt_x, txt_y, 't/tfb: ' + str(t6),
    #         fontsize = 20,
    # 		    color='white', 
    #         fontweight = 'bold',
    #         fontname = 'Consolas',
    #         transform=ax[0].transAxes)
    # # txt2.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='k')])
    
    # txt1 = ax[1].text(txt_x, txt_y, 't/tfb: ' + str(t4),
    #         fontsize = 20,
    #         color = 'white',
    #         fontweight = 'bold',
    #         fontname = 'Consolas',
    #         transform=ax[1].transAxes)
    # txt1.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='k')])    
    
    # Axis labels
    fig.text(0.5, -0.01, plane[0] + r' [x/R$_a$]', ha='center', fontsize = 20)
    fig.text(-0.02, 0.5, plane[1] + r' [y/R$_a$]', va='center', rotation='vertical', fontsize = 20)
    
    cbx = 1.08
    cby = 0.35
    # Titles
    if thing == 'Den':
        # fig.suptitle(plane + ' Density Projection - ' + title_txt, fontsize = 20)
        fig.text(cbx, cby, r'Density $\log_{10}(\rho)$ [g/cm$^2$]', fontsize = 20,
        		    color='k', fontfamily = 'monospace', rotation = 270)
    if thing == 'T':
        fig.suptitle(plane + ' Temperature Projection - ' + title_txt, fontsize = 20)
        fig.text(cbx, cby, r'Temperature $\log_{10}(T)$ [K]', fontsize = 15,
        		    color='k', fontfamily = 'monospace', rotation = 270)
    
    txt_x = 0.01
    txt_y = 0.9 
    # ax[0].text(txt_x, txt_y, '$10^6 M_\odot$', 
    #         fontsize = 20,
    #         fontfamily = 'bold',
    #         color = 'white',
    #         transform=ax[0].transAxes)
    ax.text(txt_x, txt_y, '$10^4 M_\odot$', 
            fontsize = 20,
            fontfamily = 'bold',
            color = 'white',
            transform=ax[1].transAxes)
    
from src.Utilities.finished import finished
finished()