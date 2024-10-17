#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 13:37:31 2024

@author: konstantinos
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import colorcet
import src.Utilities.prelude as c
from scipy.ndimage import uniform_filter1d 

rstar = 0.47
mstar = 0.5
Mbh = 100000
extra = 'beta1S60n1.5Compton'
simname = f'R{rstar}M{mstar}BH{Mbh}{extra}' 
pre = 'data/ef82/'
ecc = np.loadtxt(f'{pre}ecc{simname}.txt')
days = np.loadtxt(f'{pre}eccdays{simname}.txt')
mass = np.loadtxt(f'{pre}eccmass{simname}.txt')
energy = np.loadtxt(f'{pre}eccenergy{simname}.txt')
sma = np.loadtxt(f'{pre}eccsemimajoraxis{simname}.txt')
Mbh = float(Mbh)
rp = sma * (1-ecc)
nick_E_circ = float(Mbh) / (4 * rp)
angmom = np.sqrt(sma * float(Mbh) * (1 - ecc**2))
egoal = - Mbh**2/(2*angmom**2)
Rt = rstar * (float(Mbh)/mstar)**(1/3) # Msol = 1, Rsol = 1
ecirc = np.zeros_like(energy) +  Mbh/(4*Rt)
nick2 = ecirc - Mbh/(2*sma)
apocenter = Rt * (float(Mbh)/mstar)**(1/3)
radii_start = np.log10(0.4*Rt)
radii_stop = np.log10(apocenter) # apocenter
radii = np.logspace(radii_start, radii_stop, 1000)#  / Rt

#%% calc
def timer(time, radii, q, weights, Rt, goal = None, constantnumerator =  None):
    qdot = np.full_like(q, np.nan)
    t_circ = np.full_like(q, np.nan)
    
    for i in range(len(radii)):
        q_on_an_r = q.T[i]
        mask = ~np.isnan(q_on_an_r)
        qdot_temp = np.gradient(q_on_an_r[mask], time[mask])
        qdot.T[i][mask] = qdot_temp
        if type(goal) != type(None):
            goal_on_an_r = goal.T[i]
            t_circ_temp = np.divide(np.abs(q_on_an_r[mask]) - np.abs(goal_on_an_r[mask]), 
                                    qdot_temp)
            # qdot = 0 -> nan -> never going to circ if E != goal -> either 0 or inf.
            t_circ_temp = np.nan_to_num(t_circ_temp)
        elif type(constantnumerator) != type(None):
            constantnumerator_on_an_r = constantnumerator.T[i]

            t_circ_temp = np.divide(constantnumerator_on_an_r[mask], 
                                    qdot_temp)
            # qdot = 0 -> nan -> never going to circ if E != goal -> either 0 or inf.
            t_circ_temp = np.nan_to_num(t_circ_temp)
        else:
            t_circ_temp = np.divide(q_on_an_r[mask], qdot_temp)  
        t_circ.T[i][mask] = np.abs(t_circ_temp)

    # plt.figure(figsize = (3,3))
    # plt.plot(time, -qdot.T[217] * mass.T[217], 'o', c='k', markersize = 1,
    #          label = f'{radii[217]/Rt:.2} $R_\mathrm{{T}}$')
    # plt.plot(time, -qdot.T[417] * mass.T[417], 'o', c='teal', markersize = 1,
    #          label = f'{radii[417]/Rt:.2} $R_\mathrm{{T}}$')
    # plt.plot(time, -qdot.T[617] *  mass.T[617], 'o', c='sienna', markersize = 1,
    #          label = f'{radii[617]/Rt:.2} $R_\mathrm{{T}}$')
    # plt.yscale('log')
    # plt.xlabel('time [t$_\mathrm{FB}]$')
    # plt.ylabel('$\dot{\epsilon}$ [code units]')
    # plt.legend()
    
    minR = 1 * Rt
    minidx = int(np.argmin(np.abs(radii - minR)))
    maxR = 6 * Rt
    maxidx = int(np.argmin(np.abs(radii - maxR)))
    avg_range = np.arange(minidx, maxidx)
    t_circ_w = np.zeros(len(time))
    
    if type(weights) == type(None):
        for i in range(len(time)):
            for j, r in enumerate(avg_range):
                t_circ_w[i] += t_circ[i][r]  # i is time, r is radius
            t_circ_w[i] = np.divide(t_circ_w[i], len(avg_range))
    else:
        for i in range(len(time)):
            for j, r in enumerate(avg_range):
                t_circ_w[i] += t_circ[i][r] * weights[i][r] # i is time, r is radius
    return t_circ_w
if __name__ == '__main__':
    # tc_ecc_mass = timer(days, radii, q = ecc, weights = mass, Rt = Rt)
    # tc_ecc_energy = timer(days, radii, q = ecc, weights = energy, Rt = Rt)
    # tc_sma_mass = timer(days, radii, q = sma, weights = mass, Rt = Rt)
    # tc_sma_energy = timer(days, radii, q = sma, weights = energy, Rt = Rt)
    # tc_j_mass = timer(days, radii, q = angmom, weights = mass, Rt = Rt)
    # tc_j_energy = timer(days, radii, q = angmom, weights = energy, Rt = Rt)
    # tc_rp_mass = timer(days, radii, q = angmom, weights = mass, Rt = Rt)
    # tc_misunderstanding_mass = timer(days, radii, q = nick_E_circ,
    # weights = mass, Rt = Rt)
    # tc_misunderstanding_energy = timer(days, radii, q = nick_E_circ, 
    # weights = energy, Rt = Rt)
    # tc_misunderstanding_none = timer(days, radii, q = nick_E_circ, 
    # weights = None, Rt = Rt)
    # tc_mine_none = timer(days, radii, q = energy, weights = None, Rt = Rt,
    #                        goal = egoal)
    # tc_mine_none = timer(days, radii, q = energy, weights = None, Rt = Rt,
    #                        goal = ecirc)
    tc_nick_mass = timer(days, radii, q = energy, Rt=Rt, 
                         constantnumerator=nick2, weights=mass)
    tc_nick_none = timer(days, radii, q = energy, Rt=Rt, 
                         constantnumerator=nick2, weights=None)


    #%%
    plt.figure()
    # plt.plot(days, tc_ecc_mass, 'h', c = c.darkb, markersize = 3,
    #           label = 'e/$\dot{e}$ - Mass')
    # plt.plot(days, tc_ecc_energy, '^', c = c.cyan, markersize = 3,
    #           label = 'e/$\dot{e}$ - Energy')
    # plt.plot(days, tc_sma_mass, 'h', c = c.prasinaki, markersize = 3,
    #           label = r'$\alpha/\dot{\alpha}$ - Mass')
    # plt.plot(days, tc_sma_energy, '^', c = c.AEK, markersize = 3,
    #           alpha = 0.5, label = r'$\alpha/\dot{\alpha}$ - Energy')
    # plt.plot(days, tc_sma_energy, 'h', c = c.kroki, markersize = 3,
    #           alpha = 0.5, label = r'$j/\dot{j}$ - Mass')
    # plt.plot(days, tc_sma_energy, '^', c = c.reddish, markersize = 3,
    #           alpha = 0.5, label = r'$j/\dot{j}$ - Energy')
    
    plt.axvline(1.108, c='r')
    # plt.plot(days, tc_nicke_mass, 'h', c = 'k', markersize = 3,
    #           alpha = 1, label = r'$Ecirc/\dot{Ecirc}$ - Mass')
    # plt.plot(days, tc_nicke_energy, '^', c = c.reddish, markersize = 3,
    #           alpha = 0.75, label = r'$Ecirc/\dot{Ecirc}$ - Energy')
    # plt.plot(days, tc_nicke_none, 's', c = c.cyan, markersize = 3,
    #           alpha = 0.5, label = r'$Ecirc/\dot{Ecirc}$ - None')
    
    plt.plot(days, tc_nick_mass, 'h', c = 'k', markersize = 3,
              alpha = 1, label = r'Nick - Mass')
    plt.plot(days, tc_nick_none, '^', c = c.reddish, markersize = 3,
              alpha = 0.75, label = r'Nick - None')
    # plt.plot(days, tc_mine_none, 's', c = c.cyan, markersize = 3,
    #           alpha = 0.5, label = r'$Ecirc/\dot{Ecirc}$ - None')
    
    plt.legend(bbox_to_anchor = [1, 0.5, 0.1, 0.1])
    plt.yscale('log')
    plt.ylabel(r'$|t_\mathrm{circ}|$ [$t_\mathrm{FB}$]', fontsize = 13)
    plt.xlabel(r't [$t_\mathrm{FB}$]', fontsize = 13)
    Mbht = int(np.log10(Mbh))
    plt.title(f'$10^{Mbht} M_\odot$ | Circ. Time | Averaging 1-6 $R_\mathrm{{T}}$')
    #%%
    plt.figure(figsize = (3,3))
    smooth_none = uniform_filter1d(tc_nicke_none[100:], 7)
    smooth_mass = uniform_filter1d(tc_nicke_mass[100:], 7)
    
    plt.plot(days[100:], smooth_mass, '-', c = 'k', markersize = 3,
              alpha = 1, label = r'$E_\mathrm{circ}/\dot{E}_\mathrm{circ}$ - Mass')
    plt.plot(days[100:], smooth_none, '-', c = c.cyan, markersize = 3,
              alpha = 1, label = r'$E_\mathrm{circ}/\dot{E}_\mathrm{circ}$ - none')
    #plt.legend(bbox_to_anchor = [1, 0.5, 0.1, 0.1])
    plt.ylabel(r'$|t_\mathrm{circ}|$ [$t_\mathrm{FB}$]', fontsize = 13)
    plt.xlabel(r't [$t_\mathrm{FB}$]', fontsize = 13)
    plt.title(f'$10^{Mbh} M_\odot$ | Circ. Time | Averaging 1-6 $R_\mathrm{{T}}$')
    plt.ylim(0.3,8)
    plt.xlim(1.4,1.75)
    
    #%% Edot plot
    fig, ax = plt.subplots(1,1, figsize = (4,4))
    img1 = ax.pcolormesh(radii, days, edot, vmin = -1, vmax = 1,
                         cmap = 'cet_rainbow4')
    cb = fig.colorbar(img1)
    cb.set_label('edot [$t_\mathrm{FB}^{-1}$]', fontsize = 14, labelpad = 5)
    plt.xscale('log')
    plt.ylim(0.12, 1.5)
    
    plt.axvline(Rt/Rt, c = 'k')
    plt.text(Rt/Rt + 0.002, 0.3, '$R_\mathrm{T}$', 
             c = 'k', fontsize = 14)
    
    plt.axvline(0.6 * Rt/Rt, c = 'grey', ls = ':')
    plt.text(0.6 * Rt/Rt + 0.00001, 0.2, '$R_\mathrm{soft}$', 
             c = 'grey', fontsize = 14)
    # Axis labels
    fig.text(0.5, -0.01, r'r/R$_T$', ha='center', fontsize = 14)
    fig.text(-0.02, 0.5, r' Time / Fallback time $\left[ t/t_{FB} \right]$', va='center', rotation='vertical', fontsize = 14)
    ax.tick_params(axis = 'both', which = 'both', direction='in')
    #ax.set_title(r'$10^6$ M$_\odot$')
    plt.title('$ 10^5 M_\odot$')
    #%% circularization plot
    fig, ax = plt.subplots(1,1, figsize = (4,4))
    img1 = ax.pcolormesh(radii, days, t_circ, 
                         norm = colors.LogNorm(vmin = 0.1, vmax = 100),
                         cmap = 'cet_rainbow4')
    cb = fig.colorbar(img1)
    cb.set_label(r'$|t_\mathrm{circ}|$ [$t_\mathrm{FB}$]', fontsize = 14, labelpad = 5)
    plt.xscale('log')
    plt.ylim(0.12, 1.5)
    
    plt.axvline(Rt/Rt, c = 'k')
    plt.text(Rt/Rt + 0.002, 0.3, '$R_\mathrm{T}$', 
             c = 'k', fontsize = 14)
    
    plt.axvline(0.6 * Rt/Rt, c = 'grey', ls = ':')
    plt.text(0.6 * Rt/Rt + 0.00001, 0.2, '$R_\mathrm{soft}$', 
             c = 'grey', fontsize = 14)
    # Axis labels
    fig.text(0.5, -0.01, r'r/R$_a$', ha='center', fontsize = 14)
    fig.text(-0.02, 0.5, r' Time / Fallback time $\left[ t/t_{FB} \right]$', va='center', rotation='vertical', fontsize = 14)
    ax.tick_params(axis = 'both', which = 'both', direction='in')
    #ax.set_title(r'$10^6$ M$_\odot$')
    plt.title('$ 10^5 M_\odot$')

