#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 18:00:28 2024

@author: konstantinos
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import src.Utilities.prelude as c
import numba 
from tqdm import tqdm
import matplotlib.patches as patches
# Constants
mstar = 0.5  #* c.Msol_to_g     
rstar = 0.47 #* c.Rsol_to_cm
G = 1 #c.Gcgs    

@numba.njit
def solver(r0, v0, M, timesteps, what):
    dt = timesteps[1] - timesteps[0]
    # Init. arrays
    r = np.zeros((len(timesteps), 2))
    v = np.zeros((len(timesteps),  2))
    energy = np.zeros(len(timesteps))
    # Calc. constants
    rg = 2*G*M/(c.c * c.t/c.Rsol_to_cm)**2  
    # ICs
    r[0]= [r0, 0]
    v[0] = [0, -v0]
    energy[0] = 0.5*np.linalg.norm(v[0])**2 - G * M /(np.linalg.norm(r[0])-rg)
    for i in range(1, len(timesteps)):
        if what == 'leapfrog':
            # Kick
            r_mag_prev = np.linalg.norm(r[i-1])
            rhat = r[i-1]/r_mag_prev
            a_leapfrog = - rhat * G * M / (r_mag_prev-rg)**2
            v_half = v[i-1] + 0.5 * a_leapfrog * dt
            
            # Drift
            r[i] = r[i-1] + v_half * dt
            
            # Kick
            r_mag = np.linalg.norm(r[i])
            rhat = r[i]/r_mag
            a_grav = - rhat * G * M /(r_mag-rg)**2
            v[i] = v_half + 0.5 * a_grav * dt  
        energy[i] = 0.5*np.linalg.norm(v[i])**2 - G * M /(np.linalg.norm(r[i])-rg)
    return r, v, energy


# Plot results
fig, axs = plt.subplots(1,3, figsize=(8, 5), tight_layout = True)
axs = axs.flatten()
step = 10 # int(len(timesteps)*1e-5)
Ms = [1e4, 1e5, 1e6]
reccs = []
timestepss = []
plot_labels = []
for i, M in enumerate(Ms):
    rg = 2*G*M/(c.c * c.t/c.Rsol_to_cm)**2  
    Rt = rstar * (M/mstar)**(1/3)
    tfb = np.pi/np.sqrt(2) * np.sqrt(rstar**3/(G*mstar)) * np.sqrt(M/mstar) 
    timesteps = np.arange(0, 1*tfb, step = 1e-5*tfb)
    timestepss.append(timesteps/tfb)
    #    timesteps = np.arange(0, 1.1*tfb, step = 5e-5*tfb)
    

    # Most Bound
    Rp = np.linspace(Rt-rstar, Rt+rstar, num =  20)
    delta_e = G*M*rstar/Rt**2
    energies = np.linspace(-1, 1, num = 20) * delta_e
    v0s_ecc = np.sqrt(2 * (energies + G*M/(Rp-rg)))
    v0_par = np.sqrt(2*G*M/(Rt-rg))

    rpar, _, _ = solver(Rt, v0_par, M, timesteps, 'leapfrog')

    # Plotting    
    ax = axs[i]
    # m = int(np.log10(M))
    # ax.set_title(f'$M_\mathrm{{BH}}: 10^{m} M_\odot$', fontsize = 15,)
    # ax.plot(0,0, 'o', c='k', markersize = 5, markeredgecolor='k', ls = '')
    # circle = patches.Circle((0, 0), Rt, ls = '--', alpha = 0.3,
    #                         edgecolor='k', facecolor='none', linewidth=2)
    # ax.add_patch(circle)

    for ecc, r0, v0_ecc, col in tqdm(zip(energies, Rp, v0s_ecc, c.r20_palette)):
        plot_labels.append(ecc/delta_e)
        recc, _, energy = solver(r0, v0_ecc, M, timesteps, 'leapfrog')
        # recc -= rpar for distort
        reccs.append(recc)
#%% Rt constant movie movie plot
ends = np.linspace(0.01, 1, num = 200)
cols = c.r20_palette * 3
savepath = '/home/konstantinos/crossfigs/'
movname = 'constantRT'
plt.ioff()
for i, end in enumerate(ends):
    fig, axs = plt.subplots(1,3, figsize=(11, 5), tight_layout = True, dpi = 300)
    ax.plot(0,0, 'o', c='k', markersize = 5, markeredgecolor='k', ls = '')
    circle = patches.Circle((0, 0), 1, ls = '--', alpha = 0.3,
                            edgecolor='k', facecolor='none', linewidth=2)
    ax.add_patch(circle)
    for j, recc in enumerate(reccs):
        axi = int( j // (len(reccs)/len(axs)) )
        ax = axs[axi]
        idx_end = np.argmin(np.abs(end - timestepss[axi]))
        m = int( j // (len(reccs)/len(axs)) + 4 )
        
        ax.plot(recc.T[0][:idx_end:step]/Rt, recc.T[1][:idx_end:step]/Rt,
                c=cols[j], label = f'{plot_labels[j]:.2f}', ls = '-', lw = 0.75)
        ax.set_xlim(-150, 10)
        ax.set_ylim(-30, 30)
        ax.set_title(f'$M_\mathrm{{BH}}: 10^{m} M_\odot$', fontsize = 20,)

    axs[2].text(1.5, 0.2, f'Time \n {end:.2f} $t_\mathrm{{FB}}$', 
            va ='center', fontsize = 25,
            ha='center', transform=ax.transAxes)
    
    axs[2].legend(ncol = 2, bbox_to_anchor = (1.1, 1))
    axs[0].set_xlabel('X Coordinate $[R_\mathrm{T}]$', fontsize = 20)
    axs[0].set_ylabel('Y Coordinate $[R_\mathrm{T}]$', fontsize = 20)
    plt.savefig(f'{savepath}{movname}{i+1}.png')
    plt.close()

import os
os.system(f'ffmpeg -framerate 10 -i {savepath}{movname}%d.png -c:v libx264 -vf "format=yuv420p"  {savepath}mov/{movname}.mp4 -loglevel panic')
#%% Distort movie plot
ends = np.linspace(0.01, 0.1, num = 200)
cols = c.r20_palette * 3
savepath = '/home/konstantinos/crossfigs/'
movname = 'distortsmall'
plt.ioff()
for i, end in enumerate(ends):
    fig, axs = plt.subplots(1,3, figsize=(8, 5), tight_layout = True, dpi = 300)

    for j, recc in enumerate(reccs):
        axi = int( j // (len(reccs)/len(axs)) )
        ax = axs[axi]
        idx_end = np.argmin(np.abs(end - timestepss[axi]))
        m = int( j // (len(reccs)/len(axs)) + 4 )
        ax.set_title(f'$M_\mathrm{{BH}}: 10^{m} M_\odot$', fontsize = 15,)
        
        ax.plot(recc.T[0][:idx_end:step], recc.T[1][:idx_end:step],
                c=cols[j], label = f'{plot_labels[j]:.2f}', ls = '-', lw = 0.75)
    
    axs[2].text(1.4, 0.1, f'Time \n {end:.2f} $t_\mathrm{{FB}}$', 
            va ='center', fontsize = 15,
            ha='center', transform=ax.transAxes)

    axs[2].legend(ncol = 2, bbox_to_anchor = (1.1, 1))
    axs[0].set_xlabel('X Coordinate $[R_\odot]$')#'\mathrm{T}]$')
    axs[0].set_ylabel('Y Coordinate $[R_\odot]$')#'mathrm{T}]$')
    plt.savefig(f'{savepath}{movname}{i+1}.png')
    plt.close()

import os
os.system(f'ffmpeg -framerate 10 -i {savepath}{movname}%d.png -c:v libx264 -vf "format=yuv420p" -end 20 {savepath}mov/{movname}.mp4 -loglevel panic')

