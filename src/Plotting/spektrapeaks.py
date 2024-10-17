#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 18:16:05 2024

@author: konstantinos
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import colorcet

import src.Utilities.prelude as c

def loader(simname, fix):
    spectra = np.loadtxt(f'{simname}spectra{fix}.txt')
    spectra_avg = np.zeros_like(spectra[0])
    for spectrum in spectra:
        spectra_avg = np.add(spectra_avg, freqs * spectrum)
    # spectra_avg = np.divide(spectra_avg, 192)
    return spectra_avg

def plotter(spectra_avg, title, col):
    Hz_to_ev = 4.1357e-15
    ax.plot(freqs * Hz_to_ev, spectra_avg, c=col, lw = 4)

    ax.set_title(title)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e32, 5e45)
    ax.set_xlim(2e13 * Hz_to_ev, 1e21*Hz_to_ev)
    ax2 = ax.twiny()
    ax2.plot(c.c / freqs * 1e7, np.zeros(len(freqs)), c=col, lw=4)
    ax2.invert_xaxis()
    ax.set_xlabel('Energy [eV]', fontsize = 14)
    ax2.set_xscale('log')
    ax2.set_xlabel('Wavelength [nm]', fontsize = 14, labelpad = 10)
    ax.set_ylabel(r'$\nu L_\nu$ [erg/s]', fontsize = 14)
    
    
    if col == 'k':
        ypos = 7e44
        ax.axvspan(4e14*Hz_to_ev, 7e14*Hz_to_ev, alpha=0.2, color='gold')
        ax.text(3e14*Hz_to_ev, ypos, 'Opt.', fontsize = 12)
        
        ax.axvspan(7e14*Hz_to_ev, 7e16*Hz_to_ev, alpha=0.2, color='purple')
        ax.text(5e15*Hz_to_ev, ypos, 'UV', fontsize = 12)
        
        ax.axvspan(7e16*Hz_to_ev, 5e18*Hz_to_ev, alpha=0.2, color='cyan')
        ax.text(3e17*Hz_to_ev, ypos, 'Soft X', fontsize = 12)
        
        ax.axvspan(5e18*Hz_to_ev, 5e19*Hz_to_ev, alpha=0.2, color='b')
        ax.text(5.8e18*Hz_to_ev, ypos, 'Hard X', fontsize = 12)
        
    
rstar = 0.47
mstar = 0.5
extra = 'beta1S60n1.5Compton'

# 4
simname4 = f'data/blue2/R{rstar}M{mstar}BH{10000}{extra}/'
simname5 = f'data/blue2/R{rstar}M{mstar}BH{100000}{extra}/'
simname6 = f'data/blue2/R{rstar}M{mstar}BH1e+06{extra}/'


fix4 = 272
title4 = 'Spectra at Peak Luminosity | $M_\mathrm{BH} = 10^4 M_\odot | t = 1.23 t_\mathrm{FB}$ '
fix5 = 302
title5 = 'Spectra at Peak Luminosity | $M_\mathrm{BH} = 10^5 M_\odot | t = 1.11 t_\mathrm{FB}$ '
fix6 = 351
title6 = 'Spectra at Peak Luminosity | $M_\mathrm{BH} = 10^6 M_\odot | t = 0.843 t_\mathrm{FB}$ '

freqs = np.loadtxt(f'{simname4}/freqs.txt',)

spectra_avg4 = loader(simname4, fix4)
spectra_avg5 = loader(simname5, fix5)
spectra_avg6 = loader(simname6, fix6)

fig, ax = plt.subplots(1,1, figsize = (6,4))
plotter(spectra_avg4, '', 'k')
ax.text(0.75, 0.58, r'$M_\mathrm{BH} = 10^4 M_\odot$', fontsize = 13, 
        c='k', transform = ax.transAxes)
# plotter(spectra_avg5, '', c.AEK)
# ax.text(0.75, 0.65, r'$M_\mathrm{BH} = 10^5 M_\odot$', fontsize = 13, 
#         c='goldenrod', transform = ax.transAxes)
# plotter(spectra_avg6, '', 'maroon')
# ax.text(0.75, 0.72, r'$M_\mathrm{BH} = 10^6 M_\odot$', fontsize = 13, 
#         c='maroon', transform = ax.transAxes)

