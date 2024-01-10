#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 15:10:55 2023

@author: konstantinos
"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.figsize'] = [10 , 6]
plt.rcParams['axes.facecolor'] = 'whitesmoke'

plot_curves = False
residuals = True
plot_radii_sphere = True
plot_fit = False
m = 6
check = 'fid'

if plot_curves:
    # Elad Load and convert
    mat = scipy.io.loadmat('data/elad_lum.mat') 
    elad_time = mat['time']
    elad_blue = mat['L_bb']
    elad_red = mat['L_fld']
    elad_blue_topolt = np.power(10, elad_blue[0])
    elad_red_topolt = np.power(10, elad_red[0])

    # Ours Load
    daysr = np.loadtxt('data/red/alicered'+ str(m) + check + '_days.txt')
    r = np.loadtxt('data/red/alicered'+ str(m) + check + '.txt')
    daysb = np.loadtxt(f'data/blue/blue_m{m}{check}_days.txt')
    b = np.loadtxt('data/blue/bluedata_m'+ str(m) + '.txt')[2]

    # Elad Plot
    plt.plot(elad_time[0], elad_red_topolt, c = 'r')
    plt.plot(elad_time[0],elad_blue_topolt, c = 'b')

    # Our plot
    days40r = np.multiply(daysr, 40)
    days40b = np.multiply(daysb, 40)
    plt.plot(days40b, b, '--s', c='navy', markersize = 4, alpha = 0.8)
    plt.plot(days40r, r, '--', c='maroon', markersize = 4, alpha = 0.8)

    plt.yscale('log')
    plt.grid()
    plt.xlim(39, 65)
    plt.xlabel('Time [days]')
    plt.ylabel('Luminosity [erg/s]')
    plt.title('Bolometric luminosity')
    plt.savefig('Final_plot/Elad_comparison.png')
    plt.show()

    if residuals:
        y_value_r = np.zeros(len(days40r))
        y_value_b = np.zeros(len(days40b))
        for i, day in enumerate(days40r):
            y_index_r = np.argmin(np.abs(days40r[i] - elad_time[0]))
            y_value_r[i] = (elad_red_topolt[y_index_r] - r[i]) / elad_red_topolt[y_index_r]
        for i, day in enumerate(days40b):
            y_index_b = np.argmin(np.abs(days40b[i] - elad_time[0]))
            y_value_b[i] = (elad_blue_topolt[y_index_b] - b[i]) / elad_blue_topolt[y_index_b]

        plt.plot(days40r, y_value_r, c='maroon', markersize = 4)
        plt.plot(days40b, y_value_b, c='navy', markersize = 4)
        plt.grid()
        plt.xlim(39, 65)
        plt.xlabel('Time [days]')
        plt.ylabel(r'$1 - L/L^{Elad}$')
        plt.title('Bolometric luminosity: residuals')
        plt.savefig('Final_plot/residuals.png')
        plt.show()

if plot_radii_sphere:
    # Elad load 
    # mat = scipy.io.loadmat('data/elad_radii.mat')
    # elad_time = mat['x']
    # elad_amean_ph = mat['a_mean']
    # elad_gmean_ph = mat['g_mean']
    elad = np.loadtxt('data/elad_rspecial.txt')
    elad_photo_arit =  elad[0]
    elad_photo_geom =  elad[1]
    elad_therm_arit =  elad[2]
    elad_therm_geom = elad[3]

    # Our load 
    spec_radii = np.loadtxt('data/special_radii_m'+ str(m) + '.txt') #from ALICE
    days = spec_radii[0]
    days *= 40
    photo_arit = spec_radii[1]
    photo_geom = spec_radii[2]
    thermr_arit = spec_radii[3]
    thermr_geom = spec_radii[4]

    #Elad plot
    # plt.plot(elad_time[0], elad_amean_ph[0], c = 'k')
    # plt.plot(elad_time[0], elad_gmean_ph[0], c = 'magenta')
    plt.plot(days, elad_photo_arit, c = 'k')
    plt.plot(days, elad_photo_geom, c = 'magenta')
    plt.plot(days, elad_therm_arit, c = 'b')
    plt.plot(days, elad_therm_geom, c = 'r')

    # Our plot
    plt.plot(days, photo_arit, '--', color = 'black', label = 'Photosphere radius, arithmetic mean')
    plt.plot(days, photo_geom, '--', color = 'violet', label = 'Photosphere radius, geometric mean')
    plt.plot(days, thermr_arit, '--', color = 'b', label = 'Thermalization radius, arithmetic mean')
    plt.plot(days, thermr_geom, '--', color = 'r', label = 'Thermalization radius, geometric mean')
    plt.xlabel('Time (days)')
    plt.xlim(40,65)
    plt.ylim(10,1e4)
    plt.ylabel(r'Average radius [$R_\odot$]')
    plt.grid()
    plt.yscale('log')
    plt.legend(fontsize = 7)
    plt.title(r'$R_{ph}$ and $R_{therm}$')
    plt.savefig('Final_plot/radii_comparison.png')
    plt.show()

    if residuals:
        y_value_ph = (photo_geom - elad_photo_geom) / elad_photo_geom
        y_value_th = (thermr_geom - elad_therm_geom) / elad_therm_geom

        plt.plot(days, y_value_ph, c = 'magenta', markersize = 4, label = r'R$_{ph}$')
        plt.plot(days, y_value_th, c = 'r', markersize = 4, label = r'R$_{therm}$')
        plt.grid()
        plt.xlim(39, 65)
        plt.xlabel('Time [days]')
        plt.ylabel(r'$1 - R/R^{Elad}$')
        plt.legend(fontsize = 7)
        plt.title(r'$R_{ph}$ and $R_{therm}$: residuals')
        plt.savefig('Final_plot/residuals_radii.png')
        plt.show()

# if plot_fit:
#     Rsol_to_cm = 6.957e10
#     days = np.loadtxt('data/reddata_m'+ str(m) +'.txt')[0]
#     days = np.multiply(days, 40)
#     temp = np.loadtxt('data/bluedata_m'+ str(m) + '.txt')[0]
#     radius= np.loadtxt('data/bluedata_m'+ str(m) + '.txt')[1] 
#     radius = np.multiply(radius, Rsol_to_cm)

#     fig, (ax1, ax2) = plt.subplots(1, 2)
#     fig.suptitle('Fits')
#     ax1.plot(days, temp, '-o')    
#     ax1.set(xlabel = 'Times [days]', ylabel = 'Temperature [K]')
#     ax1.ticklabel_format(axis='y', style='sci', scilimits=(4,5))
#     ax2.plot(days, radius, '-o')
#     ax2.set(xlabel = 'Times [days]', ylabel = r'$\log_{10}$ Radius [cm]')
#     ax2.set_yscale('log')
#     plt.subplots_adjust(wspace=0.6)
#     plt.savefig('Final_plot/fitted_quantities.png')
#     plt.show()
