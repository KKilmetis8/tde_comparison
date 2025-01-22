#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 12:43:51 2024

@author: konstantinos
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d # does moving mean without fucking the shape up

import src.Utilities.prelude as c

def E_arrive(t, Mbh):
    '''Calculates what the energy neccecery is to come back
    at a certain time. Assumming Keplerian orbits.'''
    E = 0.5 * np.pi**2 * Mbh**2 / t**2    
    return -E**(1/3)

def Edot_fb(t, M):
    return -2/3 * (np.pi**2/2 * M**2)**(1/3) * t**(-5/3)
def Mfb(m, mstar=0.5, rstar=0.47):
    Mbh = 10**m
    Rt = rstar * (Mbh/mstar)**(1/3) 
    tfb = np.pi/np.sqrt(2) * np.sqrt(rstar**3/mstar * Mbh/mstar)
    deltaE = 2 * mstar/rstar * ((Mbh/mstar)**(1/3) + 1)

    # Data Load
    M_calli = np.load(f'data/tcirc/m_calli{m}.npy')
    E_calli = np.load(f'data/tcirc/e_calli{m}.npy')
    plt.plot(E_calli, M_calli)
    Ecirc = -Mbh/(4*Rt)
    data = np.genfromtxt(f'data/tcirc/sum{m}notspecmasked.csv', delimiter = ',')
    time = data.T[0]
    sorter = np.argsort(time)
    time = time[sorter]
    
    dMdts = np.zeros_like(time)
    dMdts_classic = np.zeros_like(time)

    # My idea, E_arrive and the analytic one
    for i in range(0, len(time)):
        arr_idx = np.argmin(np.abs( E_calli - E_arrive(time[i] * tfb, Mbh)))
        dEdt = Edot_fb(time[i] * tfb, Mbh)
        dMdE =  - M_calli[arr_idx]
        dMdts[i] = dMdE * dEdt 
        dMdts_classic[i] = - mstar/(2*tfb) * dEdt
    
    # Mass in box
    data2 = np.genfromtxt(f'data/tcirc/massinperi{m}.csv', delimiter = ',')
    time2 = data2.T[1]
    sorter = np.argsort(time2)
    time2 = time2[sorter]
    massinperi = data2.T[2][sorter]
    cumsum_mass = np.cumsum(massinperi)
    direct_mdot = np.gradient(massinperi, time2)
    
    # Steinberg & Stone
    mdot_SnS = np.load(f'data/tcirc/mdot{m}.npy')
    t_mdot_SnS = np.load(f'data/tcirc/time_for_mdot{m}.npy')
    dMdt_SnS = np.zeros(len(time))
    for i, t in enumerate(time):
        idx = np.argmin(np.abs(t_mdot_SnS - t * tfb))
        dMdt_SnS[i] = mdot_SnS[idx]
    
    # Units
    dMdts *= tfb
    dMdts_classic *= tfb
    dMdt_SnS *= tfb

    return time, dMdts, dMdts_classic, direct_mdot, time2, dMdt_SnS
t4, Mfb4, Mfbc4, box4, t42, sns = Mfb(4)
#t5, Mfb5, Mfbc5, box5 = Mfb(5)
#t6, Mfb6, Mfbc6, box6 = Mfb(6)

plt.figure(figsize = (4,3), dpi = 300)
plt.title('$\dot{M}_\mathrm{FB}$ comparison, $10^4 M_\odot$')
plt.plot(t42[box4>0], box4[box4>0], '-o', c = 'darkorange', 
          lw = 0.75, markersize = 1.5, label = 'Box')
plt.plot(t4, Mfb4, '-o', c ='royalblue',
         lw = 0.75, markersize = 1.5, label = 'dM/dE($E_\mathrm{arr}(t))$')
plt.plot(t4, sns, '--', c ='hotpink',
         lw = 0.75, markersize = 1.5, label = 'SnS')
plt.plot(t4, Mfbc4, '--', c ='k',
         lw = 0.75, markersize = 1.5, label = 'Analytic')
# plt.plot(t5, box5, '-o', c = c.AEK, 
#           lw = 0.75, markersize = 1.5, label = '5')
# plt.plot(t6, box6, '-o', c = 'maroon', 
#           lw = 0.75, markersize = 1.5, label = '6')
plt.ylabel('$\dot{M}_\mathrm{FB}$ [$M_\odot/t_\mathrm{FB}$]')
plt.xlabel('Time $[t_\mathrm{FB}]$')
plt.legend(ncols = 2, fontsize = 8)
plt.yscale('log')
plt.ylim(1e-8, 1e1)
#plt.xlim(0.2)