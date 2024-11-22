#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 10:49:14 2024

@author: konstantinos
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import colorcet
import src.Utilities.prelude as c


def tfb(Mbh, mstar = 0.5, rstar = 0.47):
    dE = -mstar/rstar * ((Mbh/mstar)**(1/3) + 1)
    a_tilde = - Mbh/(dE)
    return 2*np.pi*a_tilde**(3/2)/Mbh**(1/2)

pre = 'data/tcirc/'
Mbhs = [4, 5, 6]
fig, axs = plt.subplots(3,1, figsize = (8,6), 
                        tight_layout = True, sharex=True, sharey=True)
axs = axs.flatten()
for Mbh, ax in zip(Mbhs, axs):
    # snap, time [tfb], mean orb, mean orb - Ecirc, mean mass weighted orb, 
    # mean mass weighted orb - Ecirc

    data = np.genfromtxt(f'{pre}tcircdirect{Mbh}.csv', 
                         delimiter = ',', comments='#').T
    days =  data[1]
    orb = data[2]
    orb_minus_ec = data[3]
    mw_orb = data[4]
    mw_orb_minus_ec = data[5]
    
    orb_dot = np.gradient(orb, days)
    tcirc_orb = np.abs(orb/orb_dot)
    
    orb_minus_ec_dot = np.gradient(orb_minus_ec, days)
    tcirc_orb_minus_ec = np.abs(orb_minus_ec/orb_minus_ec_dot)
    
    mw_orb_dot = np.gradient(mw_orb, days)
    tcirc_mw_orb = np.abs(mw_orb/mw_orb_dot)
    
    mw_orb_dot_minus_ec = np.gradient(mw_orb_minus_ec, days)
    tcirc_mw_orb_minus_ec = np.abs(mw_orb_minus_ec/mw_orb_dot_minus_ec)
    
    Rt = 0.47 * (10**Mbh/0.5)**(1/3)
    ecirc = 10**Mbh / (4*Rt)
    tcirc_ec = np.abs(ecirc/orb_dot)
    tcirc_ec_mw = np.abs(ecirc/mw_orb_dot)
    
    ax.plot(days, tcirc_orb, color = 'k', 
            lw = 1, marker = 'o', markersize = 3, label = '$E/\dot{E}$')
    ax.plot(days, tcirc_orb_minus_ec, color = c.darkb,
            lw = 1, marker = 'o', markersize = 3, label = '$E - E_\mathrm{circ} / \dot{E}$')
    ax.plot(days, tcirc_mw_orb, color = c.cyan,
            lw = 1, marker = 'o', markersize = 3, label = 'MW $E/\dot{E}$')
    ax.plot(days, tcirc_mw_orb_minus_ec, color = c.prasinaki,
            lw = 1, marker = 'o', markersize = 3, label = 'MW $E - E_\mathrm{circ} / \dot{E}$' )
    ax.plot(days, tcirc_ec, color = c.AEK,
            lw = 1, marker = 'o', markersize = 3, label = '$E_\mathrm{circ} / \dot{E}$' )
    # ax.plot(days, tcirc_ec_mw, color = c.reddish,
    #        lw = 1, marker = 'o', markersize = 3, label = '$MW E_\mathrm{circ} / \dot{E}$' )
    
    ax.set_ylim(1e-2,1e3)
    ax.set_xlim(0.8)
    ax.set_yscale('log')
    # ax2.set_yscale('log')

    ax.set_title(f'$M_\mathrm{{BH}} = 10^{Mbh} M_\odot$')
ax.set_ylabel('$t_\mathrm{circ}$ [$t_\mathrm{FB}$]', fontsize = 14)
ax.set_xlabel('Time [$t_\mathrm{FB}$]', fontsize = 14)
ax.legend()

#%%
fig, axs = plt.subplots(2,3, figsize = (6,6), 
                        tight_layout = True, sharex=True, sharey=True)
cols = ['k', c.AEK, 'maroon']
axs = axs.flatten()
for Mbh, col in zip(Mbhs, cols):
    # snap, time [tfb], mean orb, mean orb - Ecirc, mean mass weighted orb, 
    # mean mass weighted orb - Ecirc

    data = np.genfromtxt(f'{pre}tcircdirect{Mbh}.csv', 
                         delimiter = ',', comments='#').T
    days =  data[1]
    orb = data[2]
    orb_minus_ec = data[3]
    mw_orb = data[4]
    mw_orb_minus_ec = data[5]
    
    orb_dot = np.gradient(orb, days)
    tcirc_orb = np.abs(orb/orb_dot)
    
    orb_minus_ec_dot = np.gradient(orb_minus_ec, days)
    tcirc_orb_minus_ec = np.abs(orb_minus_ec/orb_minus_ec_dot)
    
    mw_orb_dot = np.gradient(mw_orb, days)
    tcirc_mw_orb = np.abs(mw_orb/mw_orb_dot)
    
    mw_orb_dot_minus_ec = np.gradient(mw_orb_minus_ec, days)
    tcirc_mw_orb_minus_ec = np.abs(mw_orb_minus_ec/mw_orb_dot_minus_ec)
    
    Rt = 0.47 * (10**Mbh/0.5)**(1/3)
    ecirc = 10**Mbh / (4*Rt)
    tcirc_ec = np.abs(ecirc/orb_dot)
    tcirc_ec_mw = np.abs(ecirc/mw_orb_dot)
    
    axs[0].plot(days, tcirc_orb, color = col,
            lw = 0.5, marker = 'o', markersize = 0.5, label = f'$M_\mathrm{{BH}} = 10^{Mbh} M_\odot$' )
    axs[1].plot(days, tcirc_orb_minus_ec, color = col,
            lw = 0.5, marker = 'o', markersize = 0.5, label = f'$M_\mathrm{{BH}} = 10^{Mbh} M_\odot$' )
    axs[2].plot(days, tcirc_ec, color = col,
            lw = 0.5, marker = 'o', markersize = 0.5, label = f'$M_\mathrm{{BH}} = 10^{Mbh} M_\odot$' )
    axs[3].plot(days, tcirc_mw_orb, color = col,
            lw = 0.5, marker = 'o', markersize = 0.5, label = f'$M_\mathrm{{BH}} = 10^{Mbh} M_\odot$' )
    axs[4].plot(days, tcirc_mw_orb_minus_ec, color = col,
            lw = 0.5, marker = 'o', markersize = 0.5, label = f'$M_\mathrm{{BH}} = 10^{Mbh} M_\odot$' )
    axs[5].plot(days, tcirc_ec_mw, color = col,
            lw = 0.5, marker = 'o', markersize = 0.5, label = f'$M_\mathrm{{BH}} = 10^{Mbh} M_\odot$' )

titles = ['$E_\mathrm{orb} / \dot{E}$', 
          '$E_\mathrm{orb} - E_\mathrm{circ} / \dot{E}$',
          '$E_\mathrm{circ} / \dot{E}$',
          'MW $E_\mathrm{orb} / \dot{E}$', 
          'MW $E_\mathrm{orb} - E_\mathrm{circ} / \dot{E}$',
          'MW $E_\mathrm{circ} / \dot{E}$',]

for i, ax, title in zip(range(len(axs)), axs, titles):
    if i != len(axs):
        ax.set_ylim(0,10)
    ax.set_xlim(1)
    #ax.set_yscale('log')
    ax.set_title(title)
    
ax.set_ylabel('$t_\mathrm{circ}$ [$t_\mathrm{FB}$]', fontsize = 14)
ax.set_xlabel('Time [$t_\mathrm{FB}$]', fontsize = 14)
ax.legend()