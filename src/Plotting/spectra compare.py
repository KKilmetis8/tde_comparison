#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 15:21:57 2024

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

def plotter(spectra, spectra2, title):
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
    
    for i in range(6):
        diffs = np.abs(spectra[cardinal[i]] - spectra2[cardinal[i]]) / spectra[cardinal[i]]
        ax.plot(freqs, diffs, c = colors[i], lw = 2,
                label = labels[i])

    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e-5, 1e5)
    ax.set_xlim(2e13, 1e21)

    ax.set_title(title)
    plt.xlabel('Frequency [Hz]', fontsize = 14)
    plt.ylabel(r'$| \frac{L_\nu^\mathrm{(FID)} - L_\nu^\mathrm{(HR)} }{L_\nu^\mathrm{(FID)}} |$', fontsize = 18)
    
    ypos = 2e4
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
Mbh = 10_000 # '1e+06'
extra = 'beta1S60n1.5Compton'
simname = f'data/blue2/R{rstar}M{mstar}BH{Mbh}{extra}/'
extra = 'beta1S60n1.5ComptonHiRes'
simname2 = f'data/blue2/R{rstar}M{mstar}BH{Mbh}{extra}/'
fixes = [164, 209,] # 313]
fixes2 = [161, 210]
titles = [r'FID vs HR | 0.5 $t_\mathrm{FB}$', r'FID vs HR | 0.82 $t_\mathrm{FB}$']
spectra = loader(simname, fixes)
spectra2 = loader(simname2, fixes2)
freqs = np.loadtxt(f'{simname}/freqs.txt',)
plotter(spectra[0], spectra2[0], titles[0])