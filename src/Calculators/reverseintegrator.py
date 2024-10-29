#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 13:10:07 2024

@author: konstantinos
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import src.Utilities.prelude as c
import numba 
from tqdm import tqdm
# Constants
mstar = 0.5  #* c.Msol_to_g     
rstar = 0.47 #* c.Rsol_to_cm
G = 1 #c.Gcgs    
M = 1e6 #* c.Msol_to_g
rg = 2*G*M/(c.c * c.t/c.Rsol_to_cm)**2  
Rt = rstar * (M/mstar)**(1/3)
tfb = np.pi/np.sqrt(2) * np.sqrt(rstar**3/(G*mstar)) * np.sqrt(M/mstar) 


# Most Bound
Rp = Rt * 1
delta_e = G*M*rstar/Rp**2
energies = np.arange(-1, 1, step = 0.05) * -delta_e
v0s_ecc = np.sqrt(2 * (energies + G*M/(Rp-rg))) 

# Center of Mass
v0_par = np.sqrt(2*G*M/(Rt-rg))
timesteps = np.linspace(0, 0.01*tfb, int(1e5))
@numba.njit
def solver(r0, v0, timesteps, what):
    dt = timesteps[1] - timesteps[0]
    # Init. arrays
    r = np.zeros((len(timesteps), 2))
    v = np.zeros((len(timesteps),  2))
    energy = np.zeros(len(timesteps))

    # ICs
    r[0]= [Rt, 0]
    v[0] = [0, -v0]
    energy[0] = 0.5*v0**2 - G * M /(r0-rg)
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
        if what == 'yoshida':
            # Yoshida coefficients
            w0 = 2**(3/2) / (2 - 2**(3/2))
            w1 = 1  /  (2 - 2**(3/2))
            c1 = w1/2
            c2 =  (w0+w1)/2
            c3 = c2
            c4 = c1
            d1 = w1
            d2 = w0
            d3 = w1
            # 1
            r1 = r[i-1] + c1*v[i-1]*dt
            a1 = - r1 * G * M / (np.linalg.norm(r1)-rg)**3 
            v1 = v[i-1] + d1*a1*dt
            
            # 2 
            r2 = r1 + c2*v1*dt
            a2 = - r2 * G * M / (np.linalg.norm(r2)-rg)**3 
            v2 = v1 + d2*a2*dt
            
            # 3
            r3 = r2 + c3*v2*dt
            a3 = - r3 * G * M / (np.linalg.norm(r3)-rg)**3
            v[i] = v2 + d3*a3*dt
            
            # Final
            r[i] = r3 + c4*v[i]*dt
        energy[i] = 0.5*np.linalg.norm(v[i])**2 - G * M /(np.linalg.norm(r[i])-rg)
    return r, v, energy
rpar, vpar, energy_par = solver(Rt, v0_par, timesteps, 'leapfrog')
rpar_mag = np.linalg.norm(rpar, axis = 1)
#%%
# Save
import csv
m = int(np.log10(M))
filepath =  f'data/parabolic_orbit_{m}.csv'
with open(filepath, 'a', newline='') as file:
    writer = csv.writer(file)
    for time, r, v in zip(timesteps, rpar, vpar):
        writer.writerow([time, r[0], r[1], v[0], v[1]])
file.close()
#%%
# Plot results
fig, ax = plt.subplots(1,1, figsize=(7, 7), tight_layout = True)
step = 10 # int(len(timesteps)*1e-5)
# ax[1].plot(rpar.T[0][::step]/Rt, rpar.T[1][::step]/Rt, c='k', label = '0')
# ax[0].plot(timesteps[::step]/tfb, energy_par[::step]/delta_e, c='k',
#            ls = '-', lw = 2)
for ecc, v0_ecc, col in tqdm(zip(energies, v0s_ecc, c.c40_palette)):
    recc, _, _ = solver(Rt, v0_ecc,timesteps, 'leapfrog')
    recc_mag = np.linalg.norm(recc, axis = 1)
    #ax[2].plot(timesteps[::step]/tfb, recc_mag[::step]/Rt, c=col)
    ax.plot(recc.T[0][::step], recc.T[1][::step], 
               c=col, label = f'{ecc/delta_e:.2f}', ls = '-', lw = 2)
    # ax[0].plot(timesteps[::step]/tfb, energy[::step]/delta_e, 
    #            c=col, ls = '-', lw = 2)

ax.plot(0,0, 'o', c='k', markersize = 5, markeredgecolor='k', ls = '')
ax.legend(ncol = 2, bbox_to_anchor = (1,1))
# ax[0].set_ylabel('Energy/$\Delta E$')
# ax[0].set_xlabel('Time $[t_\mathrm{FB}]$')
ax.set_xlabel('X Coordinate $[R_\mathrm{T}]$')
ax.set_ylabel('Y Coordinate $[R_\mathrm{T}]$')

