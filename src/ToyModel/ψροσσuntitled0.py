#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:36:11 2024

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
rstar *= 1

G = 1 #c.Gcgs    
M = 1e4 #* c.Msol_to_g
rg = 2*G*M/(c.c * c.t/c.Rsol_to_cm)**2  
Rt = rstar * (M/mstar)**(1/3)
tfb = np.pi/np.sqrt(2) * np.sqrt(rstar**3/(G*mstar)) * np.sqrt(M/mstar) 
# Most Bound


# Center of Mass
@numba.njit
def solver(r0x, r0y, v0x, v0y, timesteps):
    dt = timesteps[1] - timesteps[0]
    # Init. arrays
    r = np.zeros((len(timesteps), 2))
    v = np.zeros((len(timesteps),  2))
    energy = np.zeros(len(timesteps))

    # ICs
    r[0,0]= r0x # [Rt, 0]
    r[0,1] = r0y
    v[0,0] = v0x
    v[0,1] = v0y # [0, -v0]
    energy[0] = 0.5*np.linalg.norm(v[0])**2 - G * M /(np.linalg.norm(r[0])-rg)
    for i in range(1, len(timesteps)):
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
v0_par = np.sqrt(2*G*M/(Rt-rg))
end = -0.006
rpar, vpar, energy_par = solver(Rt, 0, 0, -v0_par, timesteps = np.linspace(0, end*tfb, int(1e5)))

step = 0.2
ls = np.arange(-rstar, rstar+step*rstar, step = step*rstar)
points = []
for l in ls:
    points.append((l,0))
    points.append((0,l))
timesteps = np.linspace(0, 0.012*tfb, int(1e5))

fig, ax = plt.subplots(1,1, figsize=(7, 7), tight_layout = True)
step = 5000
for point, col in tqdm(zip(points, c.r20_palette)):
    l, r = point
    x = rpar[-1][0] + r
    y = rpar[-1][1] + l
    recc, _, energy = solver(x, y, vpar[-1][0], vpar[-1][1], timesteps)
    recc_mag = np.linalg.norm(recc, axis = 1)
    ax.plot(recc.T[0][::step], recc.T[1][::step], marker = 'o', markersize = 1.5,
               c=col, label = f'{l/rstar:.2f}', ls = '-', lw = 0.1)
ax.set_aspect('equal')

ax.plot(0,0, 'o', c='k', markersize = 5, markeredgecolor='k', ls = '')
ax.legend(ncol = 1, bbox_to_anchor = (1,1))
# plt.xlim(rpar[-1][0] - 2, rpar[-1][0] +2)
# plt.ylim(rpar[-1][1] - 2, rpar[-1][1] +2)
# plt.ylim(2.5*Rt, 3*Rt)
# plt.xlim(-1.2*Rt, -1*Rt)
# ax[0].set_ylabel('Energy/$\Delta E$')
# ax[0].set_xlabel('Time $[t_\mathrm{FB}]$')
ax.set_xlabel('X Coordinate $[R_\odot]$')
ax.set_ylabel('Y Coordinate $[R_\odot]$')

