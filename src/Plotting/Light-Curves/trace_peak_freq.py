#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 11:03:20 2025

@author: konstantinos
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import colorcet

import src.Utilities.prelude as c
import pandas as pd
def peakfinder(simname, fix, what, color, plot = False):
    spectra = np.loadtxt(f'{simname}spectra{fix}.txt')
    Rt = 0.47 * (10**Mbh / 0.5)**(1/3)


    if type(what[0]) == type(8):
        peak_freqs = np.zeros(len(what))
        for i, obs in enumerate(what): 

            spectrum_of_note = c.freqs * spectra[obs]
            peak_freqs[i] =  c.freqs[np.argmax(spectrum_of_note)]
        return peak_freqs
    
    if what == 'nick':
        peaks = []
        for i, obs in enumerate(range(c.NPIX)): 
            if color[obs] > 3: 
                spectrum_of_note = c.freqs * spectra[obs]
                peaks.append( c.freqs[np.argmax(spectrum_of_note)])        
        
        peaks = np.array(peaks)
        percentile_20 = np.percentile(peaks, 20, axis=0)
        percentile_50 = np.percentile(peaks, 50, axis=0)
        percentile_80 = np.percentile(peaks, 80, axis=0)

        return (percentile_20, percentile_50, percentile_80)

ms = [4, 5, 6]
colors = ['k', c.AEK, 'maroon']
all_days = []
all_peaks = []
all80 = []
all20 = []
all136 = []
all188 = []
for Mbh in ms:
    # if Mbh == 4:
    #     fixes = np.arange(80,344+1)
    #     # fixes = np.arange(319, 320)
    # if Mbh == 5:
    #     fixes = np.arange(132, 361+1)
    # if Mbh == 6:
    fixesstr = pd.read_csv(f'data/photosphere/sumthomp2_photocolor{Mbh}.csv').iloc[:,0][::2]
    fixes = [ int(i) for i in fixesstr]
    fixes = np.sort(fixes)

    daysstr = pd.read_csv(f'data/photosphere/sumthomp2_photocolor{Mbh}.csv').iloc[:,1][::2]
    days = [ float(i) for i in daysstr]
    days = np.sort(days)
    
    peaks20 = []
    peaks136 = []
    peaks188 = []
    colorframe = pd.read_csv(f'data/photosphere/sumthomp2_photocolor{Mbh}.csv').iloc[:,-1][::2]
    for ifix, fix in enumerate(fixes):
        pre = f'data/blue2/spectra{Mbh}/sumthomp_{Mbh}'
        color = np.array(list(map(float, colorframe.iloc[ifix].strip("[] \n").split())))
        peaks = peakfinder(pre, fix, 'nick', color)
        peaks20.append(peaks[0])
        peaks136.append(peaks[1])
        peaks188.append(peaks[2])

    peaks20 = np.array(peaks20)
    peaks136 = np.array(peaks136)
    peaks188 = np.array(peaks188)
    mask20 = peaks20 * c.Hz_to_ev > 0.1
    mask136 = peaks136 * c.Hz_to_ev > 0.1
    mask188 = peaks188 * c.Hz_to_ev > 0.1
    mask = mask136 + mask188 + mask20
    all20.append(peaks20[mask])
    all188.append(peaks188[mask])
    all136.append(peaks136[mask])
    all_days.append(days[mask])


#%%
# Plot
fig, ax = plt.subplots(1,1, figsize = (3,3))
# labels = [ ('10$^4$M$_\odot$ 10°', '10$^4$M$_\odot$ 81°'), 
#            ('10$^5$M$_\odot$ 10°', '10$^5$M$_\odot$ 81°'),
#            ('10$^6$M$_\odot$ 10°', '10$^6$M$_\odot$ 81°')]

labels = [ '10$^4$M$_\odot$', '10$^5$M$_\odot$', '10$^6$M$_\odot$',]
ax.axhspan(1.65, 3.26, alpha=0.2, color='gold') # 380 - 750 nm
ax.axhspan(3.26, 7e16 * c.Hz_to_ev, alpha=0.2, color='purple')
ax.axhspan(7e16 * c.Hz_to_ev, 5e18 * c.Hz_to_ev, alpha=0.2, color='cyan')
ev_ticks = np.array([1.65, 2.3, 3.26, 4.7, 6.5, 8.8, 12])
wavelength_ticks = 1239.8 / ev_ticks

for days, peaks20, peaks136, peaks188, co, label in zip(all_days, all20, 
                                                    all136, all188, 
                                                    colors, labels):
    ax.plot(days, peaks136 * c.Hz_to_ev, '-',  c = co, 
            label = label, alpha = 1)
    ax.fill_between(days, peaks20 * c.Hz_to_ev, peaks188 * c.Hz_to_ev, 
                    color = co, alpha = 0.3)
ev_low = 1.45
ev_high = 13
ax.set_ylim(ev_low, ev_high)

ax.set_xlabel('Time $[t_\mathrm{FB}]$')
ax.set_ylabel('Energy [eV]')

ax.set_xlim(0.8)
ax.text(0.815, 2.6, 'Visible', fontsize = 10)
ax.text(0.815, 3.45, 'UV', fontsize = 10)
ax.legend(frameon = False, fontsize = 7, ncols = 1)
ax.set_title('Peak evolution')

ax2 = ax.twinx()
ax.set_yscale('log')
ax2.set_yscale('log')

ax2.set_ylim(1239.8 / ev_low, 1239.8 / ev_high)
ax2.set_ylabel('Wavelength [nm]')
ax.set_yticks([])
ax2.set_yticks([])
import matplotlib.ticker as ticker
ax.yaxis.set_major_locator(ticker.NullLocator())
ax.yaxis.set_minor_locator(ticker.NullLocator())
ax2.yaxis.set_major_locator(ticker.NullLocator())
ax2.yaxis.set_minor_locator(ticker.NullLocator())


ax.set_yticks(ev_ticks)
ax2.set_yticks(wavelength_ticks)
ax2.axhspan(750, 380, alpha=0.2, color='gold') 

ax.set_yticklabels([f'{tick:.1f}' for tick in ev_ticks])
ax2.set_yticklabels([f'{tick:.0f}' for tick in wavelength_ticks])

