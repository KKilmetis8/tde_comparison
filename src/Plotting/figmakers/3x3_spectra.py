#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 11:19:07 2025

@author: konstantinos
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 09:13:46 2025

@author: konstantinos
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import colorcet

import src.Utilities.prelude as c

pre = 'data/bluepaper/'

fixes4 = [179, 240, 300] 
fixes5 = [227, 288, 349] #
fixes6 = [315, 379, 444] # 420->444
# zsweep = [72, 80, 76, 84, 188, 0] # x, -x, y, -y, z, -z 
zsweep = [104, 136, (152, 167), (168, 179), (180, 187), (188, 191)]#, 140]
# zsweep =  [ (20, 27), 28, (29, 35), (36, 43), (37, 42), (44, 47) ] # nside2
colors = [c.darkb, c.cyan, c.prasinaki, c.yellow, c.kroki, c.reddish, c.c99]
labels = [ r'9.6°', '30°', '42.4°', '55.3°', '68°', '81°', '90°',]
#labels =  [ r'0°', '19.4°', '26.6°', '44.1°', '66.8°', '72.9°',] #  '90°',] # nside2
#labels = ['x', '-x', 'y', '-y', 'z', '-z']
reddata4 = np.genfromtxt(f'data/red/red_richex{4}.csv', delimiter = ',').T

xticks1 = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
xticks2 = [1e-1, 1e0, 1e1, 1e2, 1e3] # 1e3]
# yticks = [1e38, 1e40, 1e42, 1e44]
yticks = [1e39, 1e41, 1e43, 1e45]
yticklabels = [ str(ytick) for ytick in yticks]
fig, ax = plt.subplots(3,3, figsize = (10,4),
                     sharey = True)
ax2s = []
for f4, f5, f6, i in zip(fixes4, fixes5, fixes6, range(3)):
    spectra4 = np.loadtxt(f'{pre}local_4spectra{f4}.txt')
    spectra5 = np.loadtxt(f'{pre}local_5spectra{f5}.txt')
    spectra6 = np.loadtxt(f'{pre}local_6spectra{f6}.txt')

    for obs,z in enumerate((zsweep)):
        try:
            l = len(z)
            spectrum4 = 0.5 * ( spectra4[ z[0] ] + spectra4[z[1]])
            spectrum5 = 0.5 * ( spectra5[ z[0] ] + spectra5[z[1]])
            spectrum6 = 0.5 * ( spectra6[ z[0] ] + spectra6[z[1]])
        except:
            spectrum4 = spectra4[ zsweep[obs] ] 
            spectrum5 = spectra5[ zsweep[obs] ] 
            spectrum6 = spectra6[ zsweep[obs] ] 
        
        Hz_to_ev = 4.1357e-15
        evs = c.freqs * Hz_to_ev
        ax[i,0].plot(evs, c.freqs * spectrum4, 
                lw = 2, c = colors[obs], label = f'{labels[obs]}')
        ax[i,1].plot(evs, c.freqs * spectrum5, 
                lw = 2, c = colors[obs], label = f'{labels[obs]}')
        ax[i,2].plot(evs, c.freqs * spectrum6, 
                lw = 2, c = colors[obs], label = f'{labels[obs]}')
    
        
        # Add wavelength
        for j in range(3):
            ax2 = ax[i,j].twiny()
            ax2.plot(c.c / c.freqs * 1e7, 
                     np.ones(len(c.freqs)), c=colors[obs], lw=4)
            ax2.set_xscale('log')
            ax2.set_xlim( 1e7 * c.c  / 5.3e19, 1e7 * c.c / 2e13)
            ax2.set_xticks(xticks2)
            ax2.invert_xaxis()

            if i != 0:
                ax2.set_xticklabels([])
                
            if i != 2:
                ax[i,j].set_xticklabels([])
            
            if i == 0 and j == 1:
                ax2.set_xlabel('Wavelength [nm]', fontsize = 14, 
                               labelpad = 10)
                
            
    # Time text
    tidx4 = np.argmin(np.abs(f4 - reddata4[0]))
    time4 = reddata4[1][tidx4]
    ax[i,2].text(0.75, 0.75, f'{time4:.2f} $t_\mathrm{{FB}}$',
                 fontsize = 12, c = 'k', transform = ax[i,2].transAxes)    

ypos = 5e43
ax[0,0].text(3e14 * Hz_to_ev, ypos, 'Vis.', fontsize = 9)
ax[0,0].text(5e15 * Hz_to_ev, ypos, 'UV', fontsize = 9)
ax[0,0].text(1.5e17 * Hz_to_ev, ypos, 'Soft X', fontsize = 9)
ax[0,0].text(5.1e18 * Hz_to_ev, ypos, 'Hard X', fontsize = 9)
rightside = [2,5,8]
bottomside = [6,7,8]
for i, oneax in enumerate(ax.flatten()):
    oneax.axvspan(4e14 * Hz_to_ev, 7e14 * Hz_to_ev, alpha=0.2, color='gold')
    oneax.axvspan(7e14 * Hz_to_ev, 7e16 * Hz_to_ev, alpha=0.2, color='purple')
    oneax.axvspan(7e16 * Hz_to_ev, 5e18 * Hz_to_ev, alpha=0.2, color='cyan')
    oneax.axvspan(5e18 * Hz_to_ev, 5.3e19 * Hz_to_ev, alpha=0.2, color='b')
    oneax.set_xscale('log')
    oneax.set_yscale('log')
    oneax.set_ylim(1e38, 3e44)
    oneax.set_xlim(2e13 * Hz_to_ev, 5.3e19 * Hz_to_ev)
    oneax.set_xticks(xticks1)
    oneax.set_yticks(yticks)
    if i !=0:
        oneax.sharey(ax.flatten()[0])
    oneax.yaxis.set_ticks_position('both')
    if i in rightside:
        ax3 = oneax.twinx()
        ax3.plot(evs, np.ones(len(evs)))
        ax3.set_yscale('log')
        ax3.yaxis.tick_right()  
        # ax3.yaxis.set_label_position("right")
        ax3.set_yticks(yticks)
        ax3.set_ylim(1e38, 3e45)
    if i not in bottomside:
        oneax.set_xticks([])
plt.tight_layout(w_pad=1)

ax[2,1].set_xlabel('Energy [eV]', fontsize = 17)
ax[1,0].set_ylabel(r'Luminosity $\nu L_\nu$ [erg/s]', fontsize = 17)
ax[0,0].set_title('$10^4 M_\odot$', fontsize = 17, y = 1.45)
ax[0,1].set_title('$10^5 M_\odot$', fontsize = 17)
ax[0,2].set_title('$10^6 M_\odot$', fontsize = 17, y = 1.45)
    # oneax.set_title(f'Spectrum {m} {fix}')
    # plt.xlabel('Frequency [Hz]', fontsize = 14)
    # plt.ylabel('Luminosity [erg/s]', fontsize = 14)
ax[1,2].legend(bbox_to_anchor = (0.7, 0.7, 1, 1), ncols = 1, fontsize = 17)




