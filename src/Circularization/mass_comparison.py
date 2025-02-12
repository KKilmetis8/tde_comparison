#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 13:12:26 2025

@author: konstantinos
"""

# Vanilla
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d # does moving mean without fucking the shape up

# Choc
import src.Utilities.prelude as c
from src.Circularization.tcirc_dmde import t_circ_dmde
from src.Circularization.taehoryucirc import taeho_circ
from src.Circularization.SnSchi import SnS_chi

# M calligraphic
# mcalt6, mcaltc6, _, _, _, _, _ = t_circ_dmde(m, 'orbdot')
t4, tc4, _, _, _, _, _ = t_circ_dmde(4, 'diss')
t5, tc5, _, _, _, _, _ = t_circ_dmde(5, 'diss')
t6, tc6, _, _, _, _, _ = t_circ_dmde(6, 'diss')


# Taeho Ryu
# ryut5, ryutc5, = taeho_circ(m, 'orbdot')
# diss_ryut5, diss_ryutc5, = taeho_circ(m, 'diss')

# # Steinberg & Stone χ
t4s, tc4s, _, _, _, _ = SnS_chi(4, 'diss')
t5s, tc5s,  _, _, _, _ = SnS_chi(5, 'diss')
t6s, tc6s,  _, _, _, _= SnS_chi(6, 'diss')

# EMR
def tc_EMR(m, rstar = 0.47, mstar = 0.5):
    # Go to CGS
    rstar *= c.Rsol_to_cm
    mstar *= c.Msol_to_g
    Mbh = 10**m * c.Msol_to_g
    
    Rt = rstar * (Mbh/mstar)**(1/3) 
    tfb = np.pi/np.sqrt(2) * (c.Gcgs * mstar / rstar**3)**(-1/2) * (Mbh/mstar)**(1/2)
    
    Ecirc = c.Gcgs * 0.5 * mstar * Mbh/(4*Rt)
    L_edd = 1.26e38 * 10**m # erg/s
    
    EMR = Ecirc / L_edd 
    return EMR / tfb
tcEMR_4 = tc_EMR(4)
tcEMR_5 = tc_EMR(5)
tcEMR_6 = tc_EMR(6)

t53 = np.array([ (i-0.7)**(-5/3) for i in np.linspace(0.7,1.8,100)])
# Time plot
fig, ax = plt.subplots(1,2, figsize = (6,4), dpi = 300, sharex = True, sharey = True, 
                       tight_layout = True)
fig.suptitle(f'Dissipation, All masses')
ax[0].set_title('$\mathcal{M}$')
ax[0].plot(t4[tc4 > 0], tc4[tc4 > 0], 
         '-o', c = 'k', lw = 0.75, markersize = 1.5, 
         label = '10$^4$M$_\odot$')
ax[0].plot(t5[tc5 > 0], tc5[tc5 > 0], 
         '-o', c = c.AEK, lw = 0.75, markersize = 1.5, 
         label = '10$^5$M$_\odot$')
ax[0].plot(t6[tc6 > 0], tc6[tc6 > 0], 
         '-o', c = 'maroon', lw = 0.75, markersize = 1.5, 
         label = '10$^6$M$_\odot$')
ax[1].set_title('SnS $\chi$')
ax[1].plot(t4s[tc4s > 0], tc4s[tc4s > 0], 
         '-o', c = 'k', lw = 0.75, markersize = 1.5, 
         label = '10$^4$M$_\odot$')
ax[1].plot(t5s[tc5s > 0], tc5s[tc5s > 0], 
         '-o', c = c.AEK, lw = 0.75, markersize = 1.5, 
         label = '10$^5$M$_\odot$')
ax[1].plot(t6s[tc6s > 0], tc6s[tc6s > 0], 
         '-o', c = 'maroon', lw = 0.75, markersize = 1.5, 
         label = '10$^6$M$_\odot$')
ax[0].plot(np.linspace(0.7,1.8,100), t53,  c = c.cyan)
ax[1].plot(np.linspace(0.7,1.8,100), t53,  c = c.cyan)
ax[0].text(1.7, 1, '5/3', c = c.cyan, rotation = -10)

# H-lines
ax[0].axhline(tcEMR_4, c = 'k', ls = '--')
ax[0].axhline(tcEMR_5, c = c.AEK, ls = '--')
ax[0].axhline(tcEMR_6, c = 'maroon', ls = '--')
ax[1].axhline(tcEMR_4, c = 'k', ls = '--')
ax[1].axhline(tcEMR_5, c = c.AEK, ls = '--')
ax[1].axhline(tcEMR_6, c = 'maroon', ls = '--')

ax[1].text(1.6, tcEMR_4 * 1.3, r'$t_\mathrm{rossi,4}$', c = 'k',)
ax[1].text(1.6, tcEMR_5 * 1.3, r'$t_\mathrm{rossi,5}$', c = c.AEK, )
ax[1].text(1.6, tcEMR_6 * 1.3, r'$t_\mathrm{rossi,6}$', c = 'maroon', )


ax[0].set_ylabel('Circularization Timescale $[t_\mathrm{FB}]$')
ax[0].set_xlabel('Time $[t_\mathrm{FB}]$')
ax[0].legend(ncols = 1, fontsize = 7, ncol = 3, frameon = False)
ax[0].set_yscale('log')
ax[1].set_yscale('log')

ax[0].set_xlim(0.68)
ax[0].set_ylim(1e-1,3e3) 
#ax[1].set_ylim(1e-2,1e6) 

