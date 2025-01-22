#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 09:34:29 2024

@author: konstantinos
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
# from src.Circularization.tcirc_dmde import t_circ_dmde
from scipy.ndimage import uniform_filter1d 

import src.Utilities.prelude as c
 
''' Let's compare Edot Ediss and Lfld '''

def peak_finder(red, t, lim = 0.45):
    start = np.argmin( np.abs(t - lim) )
    print(start)
    red_g = np.nan_to_num(red[start:])
    t_g = t[start:]
    
    peak = np.argmax(red_g)
    return red[start + peak], t[start + peak]

def Leddington(M):
    return 1.26e38 * M

def E_arrive(t, Mbh):
    '''Calculates what the energy neccecery is to come back
    at a certain time. Assumming Keplerian orbits.'''
    E = 0.5 * np.pi**2 * Mbh**2 / t**2    
    return -E**(1/3)

def t_circ_dmde(m, mstar=0.5, rstar=0.47):
    ''' '''
    Mbh = 10**m
    Rt = rstar * (Mbh/mstar)**(1/3) 
    Ecirc = -Mbh/(4*Rt)
    
    # Data Load
    M_calli = np.load(f'data/tcirc/m_calli{m}.npy')
    E_calli = np.load(f'data/tcirc/e_calli{m}.npy')
    
    data = np.genfromtxt(f'data/tcirc/sum{m}notspecmasked.csv', delimiter = ',')
    tfb = np.pi/np.sqrt(2) * np.sqrt(rstar**3/mstar * Mbh/mstar)
    time = data.T[0]
    sorter = np.argsort(time)
    time = time[sorter]
    E_orb = data.T[1][sorter]
    
    # Calculate E_orb dot
    # E_orb = uniform_filter1d(E_orb, 10)
    E_dot = np.gradient(E_orb, time) # E_fb/s
    E_dot = uniform_filter1d(E_dot, 3)
    
    tcirc = np.zeros_like(time)
    aris2 = np.zeros_like(time)
    aris1 = np.zeros_like(time)
    for i in range(0, len(time)):
        index = np.argmin(np.abs( E_calli - E_arrive(time[i] * tfb, Mbh)))
        ari1 = np.trapz(M_calli[:index], E_calli[:index]) * Ecirc # np.abs(E_orb[i])
        ari2 = E_orb[i]
        aris1[i] = ari1
        aris2[i] = ari2
        tcirc_temp =  (ari1 - ari2) / E_dot[i]
        if E_dot[i] > 0:
            tcirc[i] = tcirc[i-1]
        else:
            tcirc[i] = tcirc_temp
    # tcirc = np.abs(tcirc)
    print(m, np.median(np.abs(tcirc[-10:])))
    return time, tcirc, aris1, aris2, E_dot, E_orb, Ecirc

pre = 'data/red/'
Mbh = 4
# cols = ['k', c.AEK, 'maroon']
extra = 'beta1S60n1.5Compton'
tfb6 = np.pi/np.sqrt(2) * np.sqrt(0.47**3/0.5 * 1e6/0.5)

fig, ax = plt.subplots(1,1, figsize = (5,4), tight_layout = True, sharex=True)

#--- Edot    
t6, _, _, _, Edot6, _, _ = t_circ_dmde(Mbh)
power_converter = c.Msol_to_g * c.Rsol_to_cm**2 * c.t**(-3)
Edot6_neg = Edot6 < 0
ax.plot(t6[Edot6_neg], -Edot6[Edot6_neg]*power_converter/tfb6, color = 'k', 
        lw = 0.5, marker = 'o', markersize = 2,
        label = '$\dot{E}$')

#--- FLD
data = np.genfromtxt(f'{pre}red_richex{Mbh}.csv', delimiter = ',').T
days = data[1]
sorter = np.argsort(days)

L = data[2] / (4*np.pi) # CAREFUL

peak4, peaktime4 = peak_finder(L, days)
# Plot    
ax.plot(days[sorter], L[sorter], color=c.reddish, lw = 0.5, marker = 'o', 
        markersize = 2, label = 'FLD')
ax.plot(peaktime4, peak4, c = 'white', marker = 'X', markersize = 8, 
        markeredgecolor = c.reddish, markeredgewidth = 0.65, alpha = 0.75)

#--- Diss
data = np.genfromtxt(f'data/tcirc/sum{Mbh}diss.csv', delimiter = ',')
time = data.T[0]
sorter = np.argsort(time)
time = time[sorter]
E_diss_bound = data.T[2][sorter]
E_diss_bound_neg = E_diss_bound < 0
E_diss_bound_pos = E_diss_bound > 0

ax.plot(time[E_diss_bound_neg], -E_diss_bound[E_diss_bound_neg]*power_converter, 
        c.cyan, lw = 0.5, marker = 'o', 
        markersize = 2, label = '$E_\mathrm{diss} < 0$' )
ax.plot(time[E_diss_bound_pos], E_diss_bound[E_diss_bound_pos]*power_converter, 
        c.c98, lw = 0.5, marker = 's', 
        markersize = 2, label = '$E_\mathrm{diss} > 0$' )

# Eddington Text
ax.axhline(Leddington(10**Mbh), ls = '--', c = c.reddish)
ax.text(0.47, Leddington(10**Mbh) * 2,'$L_\mathrm{Edd}$', color = c.reddish,
            fontsize = 14 )

# Make nice
ax.set_yscale('log')
ax.set_xlim(0.15)
ax.set_ylim(1e39)

ax.set_xlabel('Time $[t_\mathrm{FB}]$', fontsize = 16)
ax.set_ylabel('Power [erg/s]', fontsize = 16)
ax.legend(loc = 'lower right', ncol = 1)

ax.set_title(f'$M_\mathrm{{BH}}$ = 10$^{Mbh}$$M_\odot$', fontsize = 14)
