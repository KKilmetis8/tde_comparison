#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:25:33 2024

@author: konstantinos
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d # does moving mean without fucking the shape up

import src.Utilities.prelude as c

#%% Funcs
def E_arrive(t, Mbh):
    '''Calculates what the energy neccecery is to come back
    at a certain time. Assumming Keplerian orbits.'''
    E = 0.5 * np.pi**2 * Mbh**2 / t**2    
    return -E**(1/3)

def t_circ_dmde(m, which, mstar=0.5, rstar=0.47):
    ''' '''
    Mbh = 10**m
    Rt = rstar * (Mbh/mstar)**(1/3)     
    # Data Load
    M_calli = np.load(f'data/tcirc/m_calli{m}.npy')
    E_calli = np.load(f'data/tcirc/e_calli{m}.npy')
    
    data = np.genfromtxt(f'data/tcirc/sum{m}notspecmasked.csv', delimiter = ',')
    tfb = np.pi/np.sqrt(2) * np.sqrt(rstar**3/mstar * Mbh/mstar)
    time = data.T[0]
    sorter = np.argsort(time)
    time = time[sorter]
    Eorb = data.T[1][sorter]
    if which == 'diss':
        data = np.genfromtxt(f'data/tcirc/sum{m}diss.csv', delimiter = ',')
        time = data.T[0]
        sorter = np.argsort(time)
        time = time[sorter]
        E_diss_bound = data.T[2][sorter] 
        Edot = E_diss_bound
    if which == 'orbdot':
        Edot = np.gradient(Eorb, time) # E_fb/s
        Edot = uniform_filter1d(Edot, 3)
    
    Ecirc = -Mbh/(4*Rt)
    tcirc = np.zeros_like(time)
    aris2 = np.zeros_like(time)
    aris1 = np.zeros_like(time)
    for i in range(0, len(time)):
        index = np.argmin(np.abs( E_calli - E_arrive(time[i] * tfb, Mbh)))
        ari1 = np.trapz(M_calli[:index], E_calli[:index]) * Ecirc # np.abs(E_orb[i])
        ari2 = Eorb[i]
        aris1[i] = ari1
        aris2[i] = ari2
        tcirc_temp =  (ari1 - ari2) / Edot[i]
        if Edot[i] > 0:
            tcirc[i] = tcirc[i-1]
        else:
            tcirc[i] = tcirc_temp
    # tcirc = np.abs(tcirc)
    print(m, np.median(np.abs(tcirc[-10:])))
    return time, tcirc, aris1, aris2, Edot, Eorb, Ecirc

if __name__ == '__main__':
    t4, tc4, a14, a24, p4, Eo4, Ec4 = t_circ_dmde(4, 'orbdot')
    t5, tc5, a15, a25, p5, Eo5, Ec5 = t_circ_dmde(5, 'orbdot')
    t6, tc6, a16, a26, p6, Eo6, Ec6 = t_circ_dmde(6, 'orbdot')
    
    plt.figure(figsize = (4,3), dpi = 300)
    plt.title('Smooth 3 | Edot')
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
    plt.ylim(1e-1,4e1)
    
    plt.text(0.55, 5, 
             f'Median \n 4 {np.median(tc4[-10:]):.2f} $t_\mathrm{{FB}}$ \n 5 {np.median(tc5[-10:]):.2f} $t_\mathrm{{FB}}$ \n 6 {np.median(tc6[-10:]):.2f} $t_\mathrm{{FB}}$')
    plt.text(0.9, 5, 
             f'Mean \n 4 {np.mean(tc4[-10:]):.2f} $t_\mathrm{{FB}}$ \n 5 {np.mean(tc5[-10:]):.2f} $t_\mathrm{{FB}}$ \n 6 {np.mean(tc6[-10:]):.2f} $t_\mathrm{{FB}}$')
    
    #%% m calli diag
    fig, ax = plt.subplots(1,1, figsize = (4,4), dpi = 300, tight_layout=True,
                           sharex = True)
    ax.set_xlim(0,2)
    ax.plot(t4, a14/Ec4, '-', c = 'k', 
             lw = 2.75, markersize = 1.5, label = '4')
    ax.plot(t5, a15/Ec5, '-', c = c.AEK, 
             lw = 1.75, markersize = 1.5, label = '5')
    ax.plot(t6, a16/Ec6, '-', c = 'maroon', 
              lw = 0.75, markersize = 1.5, label = '6')
    ax.set_yscale('log')
    ax.set_ylim(1e-10, 5e-1)
    
    ax.axhline(0.25, c = 'k', ls = ':')
    ax.text(0.2, 0.25 / 2.5, '$m_*/2$', c = 'k', va = 'center', fontsize = 12)
    # ax.text(0.2, -Ec5 * 0.25 * 1, '$m_*/2$ \n $10^5 M_\odot$', c = c.AEK, va = 'center', fontsize = 12)
    # ax.text(0.2, -Ec6 * 0.25 * 1, '$m_*/2$ \n $10^6 M_\odot$', c = 'maroon', va = 'center', fontsize = 12)
    
    # ax.axhline(-Ec5 * 0.25, c = c.AEK, ls = ':')
    # ax.axhline(-Ec6 * 0.25, c = 'maroon', ls = ':')
    ax.set_title('$\int_0^{E_\mathrm{arrive}(t)} \mathcal{M}(E) dE$')
    ax.set_ylabel('$\int_0^{E_\mathrm{arrive}(t)} \mathcal{M}(E) dE$')
    ax.set_xlabel('Time [$t_\mathrm{FB}$]')
    # test_t = np.linspace(0,2)
    # powerlaw_5over3 = test_t**(5/3) 
    # ax.plot(test_t, powerlaw_5over3, c = 'royalblue')
    #%% Show what the numerator does
    
    fig, ax = plt.subplots(3,1, figsize = (5,5), dpi = 300, tight_layout=True,
                           sharex = True)
    
    ax[0].plot(t4, a14, c='k', lw = 0.75)
    ax[0].plot(t4, a24, c='royalblue', lw = 1)
    # ax[0].axhline(0, ls = '--')
    ax[0].text(1.35, -3, '$E_\mathrm{orb}^\mathrm{(sum)}(t)$', 
               c = 'royalblue', fontsize = 12)
    ax[0].text(0.55, -6, '$E_\mathrm{circ} \int_0^{E_\mathrm{arr}(t)} \mathcal{M}(E)dE$', 
               fontsize = 12)
    ax[0].set_ylim(-10, 1)
    ax[0].set_title('$10^4 M_\odot$')
    
    ax[1].plot(t5, a15, c='k', lw = 0.75)
    ax[1].plot(t5, a25, c='royalblue', lw = 1)
    ax[1].set_ylim(-30, 3)
    ax[1].set_title('$10^5 M_\odot$')
    
    ax[2].plot(t6, a16, c='k', lw = 0.75)
    ax[2].plot(t6, a26, c='royalblue', lw = 1)
    ax[2].set_ylim(-100, 10)
    ax[2].set_title('$10^6 M_\odot$')
    ax[2].set_xlabel('Time [$t_\mathrm{FB}$]', fontsize = 12)
    #%% Is internal important?
    def energycheck(m, mstar=0.5, rstar=0.47):
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
        E_orb = data.T[1][sorter]
        return time, E_orb, data.T[2][sorter], data.T[3][sorter]
    t4, eo4, r4, ie4 = energycheck(4)
    t5, eo5, r5, ie5 = energycheck(5)
    t6, eo6, r6, ie6 = energycheck(6)
    
    fig, ax = plt.subplots(3,1, figsize = (4,6), dpi = 300, tight_layout=True)
    
    ax[0].plot(t4, eo4, c=c.cyan)
    ax[0].plot(t4, ie4, c='tomato')
    ax[0].plot(t4, r4, c=c.AEK)
    ax[0].plot(t4, eo4+ie4+r4, c='k')
    
    ax[0].text(1.45, -1.3, 'Orb', 
               c = c.cyan, fontsize = 12)
    ax[0].text(1.6, -0.25, 'Int', 
               fontsize = 12, c='tomato')
    ax[0].text(1.6, 0.15, 'Rad', 
               fontsize = 12, c=c.AEK)
    ax[0].text(1.6, -1, 'Sum', 
               fontsize = 12, c='k')
    ax[0].set_title('$10^4 M_\odot$')
    ax[0].set_ylim(-1.5, 0.4)
    
    ax[1].plot(t5, eo5, c=c.cyan)
    ax[1].plot(t5, ie5, c='tomato')
    ax[1].plot(t5, r5, c=c.AEK)
    ax[1].plot(t5, eo5+ie5+r5, c='k')
    ax[1].set_title('$10^5 M_\odot$')
    
    ax[2].plot(t6, eo6, c=c.cyan)
    ax[2].plot(t6, ie6, c='tomato')
    ax[2].plot(t6, r6, c=c.AEK)
    ax[2].plot(t6, eo6+ie6+r6, c='k')
    
    ax[2].set_title('$10^6 M_\odot$')
    ax[2].set_ylim(-50, 20)
    ax[2].set_xlabel('Time [$t_\mathrm{FB}$]', fontsize = 12)
    ax[2].set_ylabel('Energy [code]')
    #%% 4 plots diag
    fig, ax = plt.subplots(4,1, figsize = (5,6), dpi = 300, tight_layout=True,
                           sharex = True)
    ax[0].set_xlim(0,2)
    
    ax[0].plot(t4, a14, '-o', c = 'k', 
             lw = 0.75, markersize = 1.5, label = '4')
    ax[0].plot(t5, a15, '-o', c = c.AEK, 
             lw = 0.75, markersize = 1.5, label = '5')
    ax[0].plot(t6, a16, '-o', c = 'maroon', 
              lw = 0.75, markersize = 1.5, label = '6')
    # ax[0].set_yscale('log')
    # ax[0].set_ylim(1e-4, 2e2)
    ax[0].set_title('$E_\mathrm{circ} \int_0^{E_\mathrm{arrive}(t)} \mathcal{M} dE$')
    
    ax[1].plot(t4, a24, '--^', c = 'k', 
             lw = 0.75, markersize = 1.5, label = '4')
    ax[1].plot(t5, a25, '--^', c = c.AEK, 
             lw = 0.75, markersize = 1.5, label = '5')
    ax[1].plot(t6, a26, '--^', c = 'maroon', 
              lw = 0.75, markersize = 1.5, label = '6')
    #ax[1].set_yscale('log')
    ax[1].set_ylim(-30, 1)
    ax[1].set_title('$\int_0^t \dot{E}(x)dx $')
    
    def Ledd(Mbh):
        return 3.2e4 * Mbh *  c.Lsol_to_ergs
    # ax[2].axhline(Ledd(1e4), c = 'k', ls = ':')
    # ax[2].axhline(Ledd(1e5), c = c.AEK, ls = ':')
    # ax[2].axhline(Ledd(1e6), c = 'maroon', ls = ':')
    energydot_to_cgs = c.Msol_to_g * c.Rsol_to_cm**2/c.t**3
    ax[2].plot(t4, p4, '--h', c = 'k', 
             lw = 0.75, markersize = 1.5, label = '4')
    ax[2].plot(t5, p5, '--h', c = c.AEK, 
             lw = 0.75, markersize = 1.5, label = '5')
    ax[2].plot(t6, p6, '--h', c = 'maroon', 
              lw = 0.75, markersize = 1.5, label = '6')
    # ax[2].set_yscale('log')
    ax[2].set_ylim(-3e2, 10)
    ax[2].set_title('$\dot{E}(t)$ [cgs]')
    
    ax[3].plot(t4, Eo4, '--s', c = 'k', 
             lw = 0.75, markersize = 1.5, label = '4')
    ax[3].plot(t5, Eo5, '--s', c = c.AEK, 
             lw = 0.75, markersize = 1.5, label = '5')
    ax[3].plot(t6, Eo6, '--s', c = 'maroon', 
              lw = 0.75, markersize = 1.5, label = '6')
    #ax[3].set_yscale('log')
    ax[3].set_ylim(-30, 1)
    ax[3].set_title('$E_\mathrm{orb}^\mathrm{(sum)} (t) $')
    ax[-1].set_xlabel('Time [$t_\mathrm{FB}$]')
