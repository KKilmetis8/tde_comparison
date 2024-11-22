#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 11:33:51 2024

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

def taeho_circ(m, which, mstar=0.5, rstar=0.47):
    ''' '''
    Mbh = 10**m
    Rt = rstar * (Mbh/mstar)**(1/3) 
    deltaE = 2 * mstar/rstar * ((Mbh/mstar)**(1/3) + 1)
    
    # Data Load
    M_calli = np.load(f'data/tcirc/m_calli{m}.npy')
    E_calli = np.load(f'data/tcirc/e_calli{m}.npy')
    
    data = np.genfromtxt(f'data/tcirc/sum{m}notspecmasked.csv', delimiter = ',')
    tfb = np.pi/np.sqrt(2) * np.sqrt(rstar**3/mstar * Mbh/mstar)
    time = data.T[0]
    sorter = np.argsort(time)
    time = time[sorter]
    E_IE = data.T[3][sorter]
    E_Rad = data.T[2][sorter]
    E_taeho = E_Rad+E_IE
    # Calculate E_orb dot
    # if which == 'diss':
    #     data = np.genfromtxt(f'data/tcirc/sum{m}diss.csv', delimiter = ',')
    #     time = data.T[0]
    #     sorter = np.argsort(time)
    #     time = time[sorter]
    #     E_diss_bound = data.T[2][sorter] 
    #     Edot = E_diss_bound
    # if which == 'orbdot':
    #     data = np.genfromtxt(f'data/tcirc/sum{m}notspecmasked.csv', delimiter = ',')
    #     tfb = np.pi/np.sqrt(2) * np.sqrt(rstar**3/mstar * Mbh/mstar)
    #     time = data.T[0]
    #     sorter = np.argsort(time)
    #     time = time[sorter]
    #     Eorb = data.T[1][sorter]
    #     Edot = np.gradient(Eorb, time) # E_fb/s
    #     Edot = uniform_filter1d(Edot, 3)
    
    Edot = np.gradient(E_taeho, time)
    Ecirc = Mbh/(4*Rt)
    tcirc = np.zeros_like(time)
    aris2 = np.zeros_like(time)
    aris1 = np.zeros_like(time)
    for i in range(0, len(time)):
        index = np.argmin(np.abs( E_calli - E_arrive(time[i] * tfb, Mbh)))
        available_mass = np.trapz(M_calli[:index], E_calli[:index]) * Ecirc # np.abs(E_orb[i])
        tcirc[i] = available_mass/np.abs(Edot[i])
    print(m, np.median(np.abs(tcirc[-10:])))
    return time, tcirc

if __name__ == '__main__':
    t4, tc4, = taeho_circ(4)
    t5, tc5, = taeho_circ(5)
    t6, tc6, = taeho_circ(6)
    
    plt.figure(figsize = (4,3), dpi = 300)
    # plt.title('Smooth 3 | Edot')
    plt.plot(t4, tc4, '-o', c = 'k', 
             lw = 0.75, markersize = 1.5, label = '4')
    plt.plot(t5, tc5, '-o', c = c.AEK, 
             lw = 0.75, markersize = 1.5, label = '5')
    plt.plot(t6, tc6, '-o', c = 'maroon', 
              lw = 0.75, markersize = 1.5, label = '6')
    plt.ylabel('Circularization Timescale $[t_\mathrm{FB}]$')
    plt.xlabel('Time $[t_\mathrm{FB}]$')
    plt.legend(ncols = 3, fontsize = 8)
    plt.yscale('log')
    plt.xlim(0.5)
    plt.ylim(1e-2,1e3)
    plt.title('Taeho Way')
    
    plt.text(0.55, 5, 
             f'Median \n 4 {np.median(tc4[-10:]):.2f} $t_\mathrm{{FB}}$ \n 5 {np.median(tc5[-10:]):.2f} $t_\mathrm{{FB}}$ \n 6 {np.median(tc6[-10:]):.2f} $t_\mathrm{{FB}}$')
    plt.text(0.9, 5, 
             f'Mean \n 4 {np.mean(tc4[-10:]):.2f} $t_\mathrm{{FB}}$ \n 5 {np.mean(tc5[-10:]):.2f} $t_\mathrm{{FB}}$ \n 6 {np.mean(tc6[-10:]):.2f} $t_\mathrm{{FB}}$')