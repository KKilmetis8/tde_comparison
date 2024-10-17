#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 13:23:17 2024

@author: konstantinos
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import colorcet

import src.Utilities.prelude as c

def loader(simname, fixes):
    spectra = []
    for fix in fixes:
        temp_spectra = np.loadtxt(f'{simname}spectra{fix}.txt')
        spectra.append(temp_spectra)
    return spectra

def plotter(spectra, title):
    fig, ax = plt.subplots(1,1, figsize = (6,4))
    darkb = '#264653'
    cyan = '#2A9D8F'
    prasinaki = '#6a994e'
    yellow = '#E9C46A'
    kroki = '#F2A261'
    reddish = '#E76F51'
    colors = [darkb, cyan, prasinaki, yellow, kroki, reddish]
    labels = [r'-$\hat{x}$', r'$\hat{x}$', r'-$\hat{y}$', r'$\hat{y}$', 
              r'-$\hat{z}$', r'$\hat{z}$']
    
    for i in range(192):
        # ax.plot(freqs, freqs * spectra[cardinal[i]], c = colors[i], lw = 2,
        #         label = labels[i])
        ax.plot(freqs, freqs * spectra[i], lw = 2, c = rainbow_palette[i],
                label = str(i))#labels[i])
        if spectra[i][446]*freqs[446] > 1e40:
            print(i)
            print(spectra[i][446])
    ax.legend(bbox_to_anchor = (1.2,0.1,0.1,1), ncols = 10)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e39, 1e44)
    ax.set_xlim(2e13, 1e21)
    ax.set_title(title)
    plt.xlabel('Frequency [Hz]', fontsize = 14)
    plt.ylabel('Luminosity [erg/s]', fontsize = 14)
    
    ypos = 1e42
    ax.axvspan(4e14, 7e14, alpha=0.2, color='gold')
    ax.text(3e14, ypos, 'Vis.', fontsize = 12)
    
    ax.axvspan(7e14, 7e16, alpha=0.2, color='purple')
    ax.text(5e15, ypos, 'UV', fontsize = 12)
    
    ax.axvspan(7e16, 5e18, alpha=0.2, color='cyan')
    ax.text(3e17, ypos, 'Soft X', fontsize = 12)
    
    ax.axvspan(5e18, 5e19, alpha=0.2, color='b')
    ax.text(5.8e18, ypos, 'Hard X', fontsize = 12)
    
cardinal = [72, 80, 76, 84, 188, 0] # x, -x, y, -y, z, -z

rstar = 0.47
mstar = 0.5
# 4
Mbh = 100_000 #  # 
extra = 'beta1S60n1.5Compton'
simname = f'data/blue2/R{rstar}M{mstar}BH{Mbh}{extra}/'
if Mbh == 10_000:
    if 'HiRes' in extra:
        fixes = [161, 210]
        titles = [r'$10^4 M_\odot$ HiRes, 0.5 $t_\mathrm{FB}$',
                 r'$10^4 M_\odot$ HiRes, 0.82 $t_\mathrm{FB}$',]
    else:
        fixes = [164, 237, 313]
        titles = [r'$10^4 M_\odot$, 0.5 $t_\mathrm{FB}$',
                 r'$10^4 M_\odot$, 1 $t_\mathrm{FB}$', 
                 r'$10^4 M_\odot$, 1.5 $t_\mathrm{FB}$',]
elif Mbh == 100_000:
    fixes = [208, 268, 302, 365]
    titles = [r'$10^5 M_\odot$, 0.5 $t_\mathrm{FB}$',
             r'$10^5 M_\odot$, 1 $t_\mathrm{FB}$', 
             r'$10^5 M_\odot$, peak, 1.22 $t_\mathrm{FB}$', 
             r'$10^5 M_\odot$, 1.5 $t_\mathrm{FB}$',]
elif Mbh == '1e+06':
    fixes = [180, 351]
    titles = [r'$10^6 M_\odot$, 0.5 $t_\mathrm{FB}$',
             r'$10^6 M_\odot$, 0.93 $t_\mathrm{FB}$',]

spectra = loader(simname, fixes)
freqs = np.loadtxt(f'{simname}/freqs.txt',)
plotter(spectra[2], titles[2])
