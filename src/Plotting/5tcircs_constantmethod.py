#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 16:05:36 2025

@author: konstantinos
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.Circularization.tcirc_dmde import t_circ_dmde
from src.Circularization.taehoryucirc import taeho_circ
from src.Circularization.SnSchi import SnS_chi

import src.Utilities.prelude as c

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
    return EMR / tfb, Ecirc
# EMR
tcEMR_4, Ec4 = tc_EMR(4)
tcEMR_5, Ec5 = tc_EMR(5)
tcEMR_6, Ec6 = tc_EMR(6)

# Create figure
fig = plt.figure(figsize=(5,7))
gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])
# First subplot spanning the whole top half
ax1 = plt.subplot(gs[0, :])
# Four smaller subplots in the bottom half
ax2 = plt.subplot(gs[1, 0], sharex=ax1, sharey=ax1)
ax3 = plt.subplot(gs[1, 1], sharex=ax1, sharey=ax1)
ax4 = plt.subplot(gs[2, 0], sharex=ax1, sharey=ax1)
ax5 = plt.subplot(gs[2, 1], sharex=ax1, sharey=ax1)

# Prime curlyM-Diss -----------------------------------------------------------
ax1.set_title(r'$\mathcal{M}$ non-reversible')
ax1.set_yscale('log')
ax1.set_xlim(0.68,1.75)
ax1.set_ylim(1e-1,1e3) 

t4, tc4, _, _, _, _, _ = t_circ_dmde(4, 'diss')
t5, tc5, _, _, _, _, _ = t_circ_dmde(5, 'diss')
t6, tc6, _, _, _, _, _ = t_circ_dmde(6, 'diss')

ax1.plot(t4[tc4 > 0], tc4[tc4 > 0], 
         '-o', c = 'k', lw = 3.75, markersize = 1.5, 
         label = '10$^4$M$_\odot$')
ax1.plot(t5[tc5 > 0], tc5[tc5 > 0], 
         '-o', c = c.AEK, lw = 3.75, markersize = 1.5, 
         label = '10$^5$M$_\odot$')
ax1.plot(t6[tc6 > 0], tc6[tc6 > 0], 
         '-o', c = 'maroon', lw = 3.75, markersize = 1.5, 
         label = '10$^6$M$_\odot$')
ax1.set_ylabel('Circularization Timescale $[t_\mathrm{FB}]$')
ax1.set_xlabel('Time $[t_\mathrm{FB}]$')
ax1.legend(ncols = 1, fontsize = 12, ncol = 1, frameon = False)

ax1.text(0.75, 60, 
         f'Median \n 4 {np.median(tc4[-10:]):.2f} $t_\mathrm{{FB}}$ \n 5 {np.median(tc5[-10:]):.2f} $t_\mathrm{{FB}}$ \n 6 {np.median(tc6[-10:]):.2f} $t_\mathrm{{FB}}$')
print((np.mean(tc4[-5:]) - tcEMR_4) / tcEMR_4)
print((np.mean(tc5[-5:]) - tcEMR_5) / tcEMR_5)
print((np.mean(tc6[-5:]) - tcEMR_6) / tcEMR_6)

# H-lines


# curlyM - reversible ---------------------------------------------------------
t4, tc4, _, _, _, _, _ = t_circ_dmde(4, 'orbdot')
t5, tc5, _, _, _, _, _ = t_circ_dmde(5, 'orbdot')
t6, tc6, _, _, _, _, _ = t_circ_dmde(6, 'orbdot')

ax2.set_title(r'$\mathcal{M}$ reversible')
ax2.plot(t4[tc4 > 0], tc4[tc4 > 0], 
         '-o', c = 'k', lw = 0.75, markersize = 1.5, 
         label = '10$^4$M$_\odot$')
ax2.plot(t5[tc5 > 0], tc5[tc5 > 0], 
         '-o', c = c.AEK, lw = 0.75, markersize = 1.5, 
         label = '10$^5$M$_\odot$')
ax2.plot(t6[tc6 > 0], tc6[tc6 > 0], 
         '-o', c = 'maroon', lw = 0.75, markersize = 1.5, 
         label = '10$^6$M$_\odot$')
ax2.axhline(tcEMR_4, c = 'k', ls = '--')
ax2.axhline(tcEMR_5, c = c.AEK, ls = '--')
ax2.axhline(tcEMR_6, c = 'maroon', ls = '--')
# Taeho Ryu -------------------------------------------------------------------
ax3.set_title('Ryu - reversible')
ryut4, ryutc4 = taeho_circ(4)
ryut5, ryutc5 = taeho_circ(5)
ryut6, ryutc6 = taeho_circ(6)
ax3.plot(ryut4[ryutc4 > 0], ryutc4[ryutc4 > 0], '-o', c = 'k', lw = 0.75, 
         markersize = 1.5,)
ax3.plot(ryut5[ryutc5 > 0], ryutc5[ryutc5 > 0], '-o', c = c.AEK, lw = 0.75, 
         markersize = 1.5,)
ax3.plot(ryut6[ryutc6 > 0], ryutc6[ryutc6 > 0], '-o', c = 'maroon', lw = 0.75,
         markersize = 1.5,)

ryupaper_x = np.array([0.5000000000000004, 0.5114503816793896, 0.5152671755725193, 0.5801526717557253, 0.6412213740458019, 0.7290076335877864, 0.7938931297709924, 0.8435114503816796, 0.9122137404580157, 1.053435114503817, 1.1870229007633588, 1.2900763358778626, 1.4656488549618318, 1.7366412213740454, 1.5839694656488548, 1.8816793893129766, 2.0381679389312977, 2.209923664122137, 2.297709923664122, 2.442748091603053, 2.614503816793893, 2.515267175572519, 2.694656488549618, 2.797709923664122, 2.8931297709923656, 1.9694656488549618, 1.8015267175572514, 1.6564885496183206, 1.362595419847328, 0.8893129770992371, 0.965648854961832, 0.553435114503817, 0.5076335877862598, 0.5000000000000004, 1.099236641221374, 1.145038167938931, 1.2290076335877864, 2.1030534351145036, 2.171755725190839, 2.7480916030534344])
ryupaper_eta = np.array([0.00023574939199621516, 0.0004468203571895123, 0.0009013766102006907, 0.001935399302939794, 0.003042146059427333, 0.00308995960747818, 0.0041556108949685885, 0.005169556026447347, 0.008000000000000004, 0.011274318074775124, 0.015162553876590646, 0.014245621269216469, 0.01257475314703869, 0.013177034725746604, 0.013177034725746604, 0.011099861053491037, 0.01257475314703869, 0.01257475314703869, 0.012973135366835122, 0.012000000000000002, 0.0123801733983687, 0.012000000000000002, 0.01257475314703869, 0.01257475314703869, 0.01359449789842239, 0.011814313890957794, 0.0123801733983687, 0.01359449789842239, 0.013808163240518336, 0.006331387230762865, 0.00949708084614431, 0.0014168235457235254, 0.0006702305357842687, 0.00033223920391252165, 0.013808163240518336, 0.014469520346960975, 0.014696938456699072, 0.012000000000000002, 0.011814313890957794, 0.012188604546067794])
sorter = np.argsort(ryupaper_x)
ryupaper_x = ryupaper_x[sorter]
ryupaper_tc = ryupaper_x/ryupaper_eta[sorter]
# ax3.plot(ryupaper_x, ryupaper_tc, c = c.cyan, lw = 2.75, 
#          label = 'Ryu Paper')
ax3.legend(ncols = 1, fontsize = 8, ncol = 1, frameon = False)

# Steinberg & Stone χ Diss ----------------------------------------------------
ax5.set_title('SnS $\chi$ - non-reversible')
t4s, tc4s, _, _, _, _ = SnS_chi(4, 'diss')
t5s, tc5s,  _, _, _, _ = SnS_chi(5, 'diss')
t6s, tc6s,  _, _, _, _= SnS_chi(6, 'diss')

ax5.plot(t4s[tc4s > 0], tc4s[tc4s > 0], 
         '-o', c = 'k', lw = 0.75, markersize = 1.5, 
         label = '10$^4$M$_\odot$')
ax5.plot(t5s[tc5s > 0], tc5s[tc5s > 0], 
         '-o', c = c.AEK, lw = 0.75, markersize = 1.5, 
         label = '10$^5$M$_\odot$')
ax5.plot(t6s[tc6s > 0], tc6s[tc6s > 0], 
         '-o', c = 'maroon', lw = 0.75, markersize = 1.5, 
         label = '10$^6$M$_\odot$')

# Steinberg & Stone χ edot ----------------------------------------------------
ax4.set_title('SnS $\chi$ - reversible')
t4s, tc4s, _, _, _, _ = SnS_chi(4, 'orbdot')
t5s, tc5s,  _, _, _, _ = SnS_chi(5, 'orbdot')
t6s, tc6s,  _, _, _, _= SnS_chi(6, 'orbdot')

ax4.plot(t4s[tc4s > 0], tc4s[tc4s > 0], 
         '-o', c = 'k', lw = 0.75, markersize = 1.5, 
         label = '10$^4$M$_\odot$')
ax4.plot(t5s[tc5s > 0], tc5s[tc5s > 0], 
         '-o', c = c.AEK, lw = 0.75, markersize = 1.5, 
         label = '10$^5$M$_\odot$')
ax4.plot(t6s[tc6s > 0], tc6s[tc6s > 0], 
         '-o', c = 'maroon', lw = 0.75, markersize = 1.5, 
         label = '10$^6$M$_\odot$')


yticks = [1e-2, 1e0, 1e2,] # 1e3]
for ax in [ax1, ax2, ax3, ax4, ax5]:
    ax.set_yticks(yticks)
    
    ax.axhline(tcEMR_4, c = 'k', ls = '--')
    ax.axhline(tcEMR_5, c = c.AEK, ls = '--')
    ax.axhline(tcEMR_6, c = 'maroon', ls = '--')
plt.tight_layout()
