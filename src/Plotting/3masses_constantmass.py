#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 17:11:42 2025

@author: konstantinos
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:58:41 2024

@author: konstantinos

compare all 6 tcircs
"""

# Vanilla
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d # does moving mean without fucking the shape up
import matplotlib.gridspec as gridspec

# Choc
import src.Utilities.prelude as c
from src.Circularization.tcirc_dmde import t_circ_dmde
from src.Circularization.taehoryucirc import taeho_circ
from src.Circularization.SnSchi import SnS_chi

# Create figure
fig = plt.figure(figsize=(5,7))
gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])
# First subplot spanning the whole top half
ax1 = plt.subplot(gs[0, :])
# Four smaller subplots in the bottom half
ax2 = plt.subplot(gs[1, 0])
ax3 = plt.subplot(gs[1, 1])

ms = [4, 5, 6]
axs = [ax2, ax1, ax3]
ylims = [ (5e-1, 4e2), (5e-1, 1e3), (1e-2, 1e2)]
colors = [c.c94, c.c91, c.c97, c.c99, c.c92, c.cyan]
for m, ax, ylim in zip(ms, axs, ylims):
    if m == 5:
        lw = 2
    else:
        lw = 1.2
    # M calligraphic
    mcal_t6, mcal_tc6, _, _, _, _, _ = t_circ_dmde(m, 'orbdot')
    diss_mcal_t6, diss_mcal_tc6, _, _, _, _, _ = t_circ_dmde(m, 'diss')
    # Taeho Ryu
    ryu_t5, ryu_tc5, = taeho_circ(m)    
    # Steinberg & Stone χ
    sns_t6, sns_tc6, _, _, _, _= SnS_chi(m, 'diss')
    my_sns_t6, my_sns_tc6, _, _ , _ ,_ = SnS_chi(m, 'orbdot')

    ax.set_title(f'$10^{m} M_\odot$')
    
    ax.plot(diss_mcal_t6[diss_mcal_tc6 > 0], diss_mcal_tc6[diss_mcal_tc6 > 0], 
             '-', c = colors[0], lw = lw, markersize = 1.5, 
             label = ' $\mathcal{M}$ non-PdV')
    ax.plot(mcal_t6[mcal_tc6 > 0], mcal_tc6[mcal_tc6 > 0 ], 
             '-', c = colors[1],  lw = lw, markersize = 1.5, 
             label = '$\mathcal{M}$ PdV')
    
    
    # plt.plot(diss_ryu_t5[diss_ryu_tc5 > 0], diss_ryu_tc5[diss_ryu_tc5 > 0 ], 
    #          '--s', c = c.yellow,  lw = 0.75, markersize = 1.5, 
    #          label = 'Ryu $E_\mathrm{diss}$')
    ax.plot(ryu_t5[ryu_tc5 > 0], ryu_tc5[ryu_tc5 > 0], 
             '-', c = colors[2], lw = lw, markersize = 1.5, 
             label = 'Ryu')
    
    ax.plot(sns_t6, sns_tc6, '-', c = colors[3], 
              lw = lw, markersize = 1.5, label = r'$\chi$ non-PdV')
    ax.plot(my_sns_t6, my_sns_tc6, '-', c = colors[4], 
             lw = lw, markersize = 1.5, label = r'$\chi$ PdV')
    ax.set_xlim(0.75)
    ax.set_yscale('log')
    ax.set_ylim(ylim[0], ylim[1]) 
    if m == 5:
        ryupaper_x = np.array([0.5000000000000004, 0.5114503816793896, 0.5152671755725193, 0.5801526717557253, 0.6412213740458019, 0.7290076335877864, 0.7938931297709924, 0.8435114503816796, 0.9122137404580157, 1.053435114503817, 1.1870229007633588, 1.2900763358778626, 1.4656488549618318, 1.7366412213740454, 1.5839694656488548, 1.8816793893129766, 2.0381679389312977, 2.209923664122137, 2.297709923664122, 2.442748091603053, 2.614503816793893, 2.515267175572519, 2.694656488549618, 2.797709923664122, 2.8931297709923656, 1.9694656488549618, 1.8015267175572514, 1.6564885496183206, 1.362595419847328, 0.8893129770992371, 0.965648854961832, 0.553435114503817, 0.5076335877862598, 0.5000000000000004, 1.099236641221374, 1.145038167938931, 1.2290076335877864, 2.1030534351145036, 2.171755725190839, 2.7480916030534344])
        ryupaper_eta = np.array([0.00023574939199621516, 0.0004468203571895123, 0.0009013766102006907, 0.001935399302939794, 0.003042146059427333, 0.00308995960747818, 0.0041556108949685885, 0.005169556026447347, 0.008000000000000004, 0.011274318074775124, 0.015162553876590646, 0.014245621269216469, 0.01257475314703869, 0.013177034725746604, 0.013177034725746604, 0.011099861053491037, 0.01257475314703869, 0.01257475314703869, 0.012973135366835122, 0.012000000000000002, 0.0123801733983687, 0.012000000000000002, 0.01257475314703869, 0.01257475314703869, 0.01359449789842239, 0.011814313890957794, 0.0123801733983687, 0.01359449789842239, 0.013808163240518336, 0.006331387230762865, 0.00949708084614431, 0.0014168235457235254, 0.0006702305357842687, 0.00033223920391252165, 0.013808163240518336, 0.014469520346960975, 0.014696938456699072, 0.012000000000000002, 0.011814313890957794, 0.012188604546067794])
        sorter = np.argsort(ryupaper_x)
        ryupaper_x = ryupaper_x[sorter]
        ryupaper_tc = ryupaper_x/ryupaper_eta[sorter]
        ax.plot(ryupaper_x, ryupaper_tc, c = colors[5], lw = 2.75, 
                 label = 'Ryu Paper')
        
        ax.set_ylabel(f'Circularization Timescale $[t_\mathrm{{FB}}]$')
        ax.set_xlabel('Time $[t_\mathrm{FB}]$')
        ax.legend(ncols = 2, fontsize = 10, frameon = False)
        # ax.set_ylim(1e-2, 7e2) 
