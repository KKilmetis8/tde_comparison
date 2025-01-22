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

# Choc
import src.Utilities.prelude as c
from src.Circularization.tcirc_dmde import t_circ_dmde
from src.Circularization.taehoryucirc import taeho_circ
from src.Circularization.SnSchi import SnS_chi

m = 6
# M calligraphic
mcal_t6, mcal_tc6, _, _, _, _, _ = t_circ_dmde(m, 'orbdot')
diss_mcal_t6, diss_mcal_tc6, _, _, _, _, _ = t_circ_dmde(m, 'diss')

# Taeho Ryu
ryu_t5, ryu_tc5, = taeho_circ(m, 'orbdot')
diss_ryu_t5, diss_ryu_tc5, = taeho_circ(m, 'diss')

# Steinberg & Stone χ
sns_t6, sns_tc6, _, _, _, _= SnS_chi(m, 'diss')
my_sns_t6, my_sns_tc6, _, _ , _ ,_ = SnS_chi(m, 'orbdot')

# Time plot
plt.figure(figsize = (4,3), dpi = 300)
plt.title(f'Gran Comparison, $10^{m} M_\odot$')

plt.plot(diss_mcal_t6[diss_mcal_tc6 > 0], diss_mcal_tc6[diss_mcal_tc6 > 0], 
         '-o', c = c.reddish, lw = 0.75, markersize = 1.5, 
         label = ' $\mathcal{M}$ $E_\mathrm{diss}$')
plt.plot(mcal_t6[mcal_tc6 > 0], mcal_tc6[mcal_tc6 > 0 ], 
         '--s', c = c.yellow,  lw = 0.75, markersize = 1.5, 
         label = '$\mathcal{M}$ $\dot{E}_\mathrm{orb}$')


# plt.plot(diss_ryu_t5[diss_ryu_tc5 > 0], diss_ryu_tc5[diss_ryu_tc5 > 0 ], 
#          '--s', c = c.yellow,  lw = 0.75, markersize = 1.5, 
#          label = 'Ryu $E_\mathrm{diss}$')
plt.plot(ryu_t5[ryu_tc5 > 0], ryu_tc5[ryu_tc5 > 0], 
         '-o', c = c.prasinaki, lw = 0.75, markersize = 1.5, 
         label = 'Ryu $\dot{E}_\mathrm{rad + gas}$')

plt.plot(sns_t6, sns_tc6, '-o', c = c.cyan, 
          lw = 0.75, markersize = 1.5, label = 'SnS $\chi$ $E_\mathrm{diss}$')
plt.plot(my_sns_t6, my_sns_tc6, '--s', c = c.darkb, 
         lw = 0.75, markersize = 1.5, label = 'SnS $\chi$ $\dot{E}_\mathrm{orb}$')
plt.ylabel(f'Circularization Timescale $[t_\mathrm{{FB}}]$, $10^{m}, M_\odot$')
plt.xlabel('Time $[t_\mathrm{FB}]$')
plt.legend(ncols = 1, fontsize = 8, bbox_to_anchor = [1,1,0,0])
plt.yscale('log')
plt.xlim(0.8)
plt.ylim(1e-2, 3e1) 
