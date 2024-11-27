#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 18:09:50 2024

@author: konstantinos
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import colorcet

import src.Utilities.prelude as c

rstar = 0.47
mstar = 0.5

Mbh = '1e+06'
extra = 'beta1S60n1.5Compton'
simname6 = f'data/red/redR{rstar}M{mstar}BH{Mbh}{extra}'
red6 = np.loadtxt(f'{simname6}_lums.txt')
t6 = np.loadtxt(f'{simname6}_days.txt')

Mbh = 100000
extra = 'beta1S60n1.5Compton'
simname5 = f'data/red/redR{rstar}M{mstar}BH{Mbh}{extra}'
red5 = np.loadtxt(f'{simname5}_lums.txt')
t5 = np.loadtxt(f'{simname5}_days.txt')

# Elads 
simname5 = f'data/red/R{rstar}M{mstar}BH{Mbh}{extra}'
red5_elad = np.loadtxt(f'{simname5}_eladred_191to251.txt')
t5_elad = np.loadtxt(f'{simname5}_eladreddays_191to251.txt')

Mbh = 10000
extra = 'beta1S60n1.5Compton'
simname4 = f'data/red/redR{rstar}M{mstar}BH{Mbh}{extra}' 
red4 = np.loadtxt(f'{simname4}_lums.txt')
t4 = np.loadtxt(f'{simname4}_days.txt')



extra = 'beta1S60n1.5ComptonHiRes'
simname4h = f'data/red/redR{rstar}M{mstar}BH{Mbh}{extra}' 
red4h = np.loadtxt(f'{simname4h}_lums.txt')
t4h = np.loadtxt(f'{simname4h}_days.txt')

extra = 'beta1S60n1.5ComptonRes20'
simname4s = f'data/red/redR{rstar}M{mstar}BH{Mbh}{extra}' 
red4s = np.loadtxt(f'{simname4s}_lums.txt')
t4s = np.loadtxt(f'{simname4s}_days.txt')



#%%
fig, ax = plt.subplots(1,1, figsize = (4,4))
ax.plot(t4, red4, c = 'k', lw = 2, label = r'$10^4 M_\odot$')
ax.plot(t4h, red4h, c = 'k', lw = 1, ls = '--', label = r'$10^4 M_\odot$ | HR', )
ax.plot(t4s, red4s, c = 'k', lw = 1, ls = ':', label = r'$10^4 M_\odot$ | SHR',)
ax.plot(t5, red5, c = c.AEK, lw = 2, label = r'$10^5 M_\odot$')
ax.plot(t5_elad, red5_elad, c = 'b', lw = 1, ls = '--', label = r'$10^5 M_\odot$ real' )
ax.plot(t6, red6, c = 'maroon', lw = 2, label = r'$10^6 M_\odot$')
plt.yscale('log')
plt.ylim(1e39, 1e46)
plt.xlim(0.4)
plt.title(f'FLD Luminosity')
plt.xlabel('Time [$t_\mathrm{FB}$]', fontsize = 14)
plt.ylabel('Luminosity [erg/s]', fontsize = 14)
plt.legend()