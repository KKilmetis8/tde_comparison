#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 10:58:18 2024

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

def Mdot_fb(t, tfb, mstar = 0.5):
    return mstar/(3*tfb) * (t/tfb)**(-5/3)

def Leddington(M):
    return 1.26e38 * M


def SnS_chi(m, which, mstar=0.5, rstar=0.47):
    Mbh = 10**m
    Rt = rstar * (Mbh/mstar)**(1/3) 
    deltaE = 2 * mstar/rstar * ((Mbh/mstar)**(1/3) + 1)
    tfb = np.pi/np.sqrt(2) * np.sqrt(rstar**3/mstar * Mbh/mstar)

    # Data Load
    M_calli = np.load(f'data/tcirc/m_calli{m}.npy')
    E_calli = np.load(f'data/tcirc/e_calli{m}.npy')
    Ecirc = -Mbh/(4*Rt)
    
    if which == 'diss':
        data = np.genfromtxt(f'data/tcirc/sum{m}diss.csv', delimiter = ',')
        time = data.T[0]
        sorter = np.argsort(time)
        time = time[sorter]
        E_diss_bound = data.T[2][sorter] 
        Edot = -E_diss_bound
        
    if which == 'orbdot':
        data = np.genfromtxt(f'data/tcirc/sum{m}notspecmasked.csv', delimiter = ',')
        time = data.T[0]
        sorter = np.argsort(time)
        time = time[sorter]
        Eorb = data.T[1][sorter]
        Edot = np.gradient(Eorb, time) / tfb # E_fb/s
        Edot = uniform_filter1d(Edot, 3)
        
    tcirc = np .zeros_like(time)
    chis = np.zeros_like(time)
    dMdts = np.zeros_like(time)
    for i in range(0, len(time)):
        arr_idx = np.argmin(np.abs( E_calli - E_arrive(time[i] * tfb, Mbh)))
        
        dEdt = Edot_fb(time[i] * tfb, Mbh)
        dMdts[i] =  M_calli[arr_idx] * dEdt # Mdot_fb(time[i]*tfb, tfb)#  dMdE_avg * dEdt
        chi = - Edot[i] / (Ecirc * dMdts[i])
        # print(chi)
        chis[i] = chi
        tcirc_temp = time[i] / chi
        if Edot[i] > 0:
            tcirc[i] = tcirc[i-1]
        else:
            tcirc[i] = tcirc_temp
    return time, tcirc, chis, dMdts, -Edot, dMdts*Ecirc


if __name__ == '__main__':
    t6, tc6, chis, dMdT, ari, par = SnS_chi(6, 'diss')
    my_t6, my_tc6, my_chis, my_dMdT, my_ari, my_par = SnS_chi(6, 'orbdot')
    
    # Time plot
    plt.figure(figsize = (4,3), dpi = 300)
    plt.title('SnS $|$ $t_\mathrm{circ} = t / \chi $')
    plt.plot(t6, tc6, '-o', c = c.cyan, 
              lw = 0.75, markersize = 1.5, label = '$E_\mathrm{diss}$')
    plt.plot(my_t6, my_tc6, '--s', c = c.darkb, 
             lw = 0.75, markersize = 1.5, label = '$\dot{E}_\mathrm{orb}$')
    plt.ylabel('Circularization Timescale $[t_\mathrm{FB}]$')
    plt.xlabel('Time $[t_\mathrm{FB}]$')
    plt.legend(ncols = 3, fontsize = 8)
    plt.yscale('log')
    plt.xlim(0.2)
    plt.ylim(1e-4,8e2)
    #%%
    
    # Diss diagnostic
    tfb6 = np.pi/np.sqrt(2) * np.sqrt(0.47**3/0.5 * 1e6/0.5)
    fig, axs = plt.subplots(2,1, figsize = (5,6), dpi = 300, tight_layout = True,
                            sharex = True)
    axs = axs.flatten()
    ax = axs[0]
    fig.suptitle('Circ Efficiency')
    ax.set_title('Dissipation from RICH')
    sec_to_yr = 1 / (c.day_to_sec * 365)
    #t6 *= tfb6 * c.t * sec_to_yr
    ax.plot(t6, chis, '-o', c = 'royalblue', 
              lw = 0.75, markersize = 1.5, label = '$\chi$')
    ax2 = ax.twinx()
    
    ax2.axhline(Leddington(10**6), ls = '--', c = 'k')
    ax2.text(0.21, Leddington(10**6) * 2,'$L_\mathrm{Edd}$', color = 'k',
                fontsize = 12 )
    
    power_converter = c.Msol_to_g * c.Rsol_to_cm**2 * c.t**(-3)
    ax2.plot(t6, ari * power_converter, 
             '-o', c = 'firebrick', lw = 0.75, markersize = 1.5, 
             label = '$E_\mathrm{diss}$')
    ax2.plot(t6, par * power_converter, 
             '-o', c = 'yellowgreen', lw = 0.75, markersize = 1.5,
             label = '$E_\mathrm{circ} \dot{M}$')
    
    ax.set_ylabel('$\chi = -E_\mathrm{diss}/E_\mathrm{circ} \dot{M}_\mathrm{fb}$')
    ax2.set_ylabel('Power [erg/s]') # '$\dot{M}_\mathrm{FB}$ [$M_\odot/t_\mathrm{FB}$]')
    # ax.set_xlabel('Time $[t_\mathrm{FB}]$')
    
    ax2.spines['left'].set_color('royalblue')
    ax.tick_params(axis='y', colors='royalblue')
    ax2.spines['right'].set_color('k')
    ax2.tick_params(axis='y', colors='k')
    ax2.legend(loc = 'lower left')
    ax2.set_yscale('log')
    ax.set_xlim(0.2)
    ax.set_yscale('log')
    
    # Without diss diagnostic
    ax = axs[1]
    ax.set_title('$\dot{E}_\mathrm{orb}$')
    ax.plot(my_t6, my_chis, '-o', c = 'royalblue', 
              lw = 0.75, markersize = 1.5, label = '$\chi$')
    ax.set_yscale('log')
    ax2 = ax.twinx()
    
    ax2.axhline(Leddington(10**6), ls = '--', c = 'k')
    ax2.text(0.21, Leddington(10**6) * 3,'$L_\mathrm{Edd}$', color = 'k',
                fontsize = 12 )
    
    power_converter = c.Msol_to_g * c.Rsol_to_cm**2 * c.t**(-3)
    ax2.plot(my_t6, my_ari * power_converter, 
             '-o', c = 'firebrick', lw = 0.75, markersize = 1.5, 
             label = '$\dot{E}_\mathrm{orb}$')
    ax2.plot(my_t6, my_par * power_converter, 
             '-o', c = 'yellowgreen', lw = 0.75, markersize = 1.5,
             label = '$E_\mathrm{circ} \dot{M}$')
    
    ax.set_ylabel('$\chi = -\dot{E}_\mathrm{orb}/E_\mathrm{circ} \dot{M}_\mathrm{fb}$')
    ax2.set_ylabel('Power [erg/s]') # '$\dot{M}_\mathrm{FB}$ [$M_\odot/t_\mathrm{FB}$]')
    ax.set_xlabel('Time $[t_\mathrm{FB}]$')
    
    ax2.spines['left'].set_color('royalblue')
    ax.tick_params(axis='y', colors='royalblue')
    ax2.spines['right'].set_color('k')
    ax2.tick_params(axis='y', colors='k')
    ax2.legend(loc = 'lower center')
    ax2.set_yscale('log')
    ax2.set_ylim(1e37, 1e48)
    ax.set_xlim(0.2)
    # ax.set_ylim(10, 3000)
    # ax.set_yscale('log')