#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:24:14 2024

@author: konstantinos
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 16:02:20 2024

@author: konstantinos
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from src.Utilities.isalice import isalice
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
alice, plot = isalice()
import src.Utilities.prelude as c
from src.Utilities.parser import parse
#%% Choose parameters -----------------------------------------------------------------
if alice:
    pre = '/home/kilmetisk/data1/TDE/'
    args = parse()
    sim = args.name
    mstar = args.mass
    rstar = args.radius
    Mbh = args.blackhole
    fixes = np.arange(args.first, args.last + 1)
else:
    m = 4
    Mbh = 10**m
    pre = f'{m}/'
    fixes = [80, 136, 164, 199, 348]
    mstar = 0.5
    rstar = 0.47
    deltaE = mstar/rstar * ((Mbh/mstar)**(1/3) + 1)
    if m == 4:
        change = 80
    if m == 5:
        change = 132
    if m == 6:
        change = 180

# Do it --- -----------------------------------------------------------------
rg = 2*float(Mbh)/(c.c * c.t/c.Rsol_to_cm)**2
Rt = rstar * (Mbh/mstar)**(1/3) 
Ts = np.zeros(len(fixes))
times = np.zeros(len(fixes))
up = np.zeros(len(fixes))
down = np.zeros(len(fixes))
maxes = np.zeros(len(fixes))
supermaxes = np.zeros(len(fixes))
superomegamaxes = np.zeros(len(fixes))
mines = np.zeros(len(fixes))
for idx_s, snap in tqdm(enumerate(fixes)):
    # Load data
    if alice:
        X = np.load(f'{pre}{sim}/snap_{snap}/CMx_{snap}.npy')
        Y = np.load(f'{pre}{sim}/snap_{snap}/CMy_{snap}.npy')
        Z = np.load(f'{pre}{sim}/snap_{snap}/CMz_{snap}.npy')
        VX = np.load(f'{pre}{sim}/snap_{snap}/Vx_{snap}.npy')
        VY = np.load(f'{pre}{sim}/snap_{snap}/Vy_{snap}.npy')
        VZ = np.load(f'{pre}{sim}/snap_{snap}/Vz_{snap}.npy')
        T = np.load(f'{pre}{sim}/snap_{snap}/T_{snap}.npy')
        Vol = np.load(f'{pre}{sim}/snap_{snap}/Vol_{snap}.npy')
        day = np.loadtxt(f'{pre}{sim}/snap_{snap}/tbytfb_{snap}.txt')
        m = int(np.log10(float(Mbh)))
    else:
        X = np.load(f'{pre}{snap}/CMx_{snap}.npy')
        Y = np.load(f'{pre}{snap}/CMy_{snap}.npy')
        Z = np.load(f'{pre}{snap}/CMz_{snap}.npy')
        VX = np.load(f'{pre}{snap}/Vx_{snap}.npy')
        VY = np.load(f'{pre}{snap}/Vy_{snap}.npy')
        VZ = np.load(f'{pre}{snap}/Vz_{snap}.npy')
        T = np.load(f'{pre}{snap}/IE_{snap}.npy')
        Den = np.load(f'{pre}{snap}/Den_{snap}.npy')
        Vol = np.load(f'{pre}{snap}/Vol_{snap}.npy')
        day = np.loadtxt(f'{pre}{snap}/tbytfb_{snap}.txt')
        
    Orb = 0.5 * (VX**2 + VY**2 + VZ**2) - Mbh/(np.sqrt(X**2+Y**2+Z**2) - rg)
    bound_mask = Orb < 0
    times[idx_s] = day
    Mass = Den * Vol
    T *= Mass
    T *= c.Msol_to_g * c.Rsol_to_cm**2/c.t**2
    Ts[idx_s] = np.sum(T[bound_mask])#, weights = Mass[bound_mask])
    percentiles = np.percentile(T[bound_mask], [0, 10, 90, 99, 99.9, 100], 
                                method = 'inverted_cdf', 
                                weights = Mass[bound_mask])
    mines[idx_s] = percentiles[0]
    down[idx_s] = percentiles[1]
    up[idx_s] = percentiles[2]
    maxes[idx_s] = percentiles[3]
    supermaxes[idx_s] = percentiles[4]
    superomegamaxes[idx_s] = percentiles[5]
#%%
plt.figure( figsize=(4,3), dpi = 300)
plt.plot(times, superomegamaxes, '-o', c='darkslateblue', label = 'max',
         lw = 0.75, markersize = 1.5,)
plt.plot(times, supermaxes, '-o', c='b', label = '99.9',
         lw = 0.75, markersize = 1.5,)
plt.plot(times, maxes, '-o', c='cornflowerblue', label = '99',
         lw = 0.75, markersize = 1.5,)
plt.plot(times, up, '-o', c=c.cyan, label = '90',
         lw = 0.75, markersize = 1.5,)
plt.plot(times, Ts, '-o', c='k', label = 'mean',
         lw = 0.75, markersize = 1.5,)
plt.plot(times, down, '-o', c='tomato', label = '10', 
         lw = 0.75, markersize = 1.5,)
# plt.plot(times, mines/Ts[0], '-o', c='maroon', label = 'min',
#          lw = 0.75, markersize = 1.5,) 
plt.legend(ncols = 1, fontsize = 8, bbox_to_anchor = [1,1,0,0])

plt.yscale('log')
plt.xlabel('Time [$t_\mathrm{FB}$]')
plt.ylabel('MW Mean Rad of bound material [erg]')
plt.title('$10^4 M_\odot$ ')

#%%
surface_density = 1e13 # g/cm2
rg = 2 * c.G * 5e3 * c.Msol_to_g / c.c**2
z_guess = 2 * rg # cm
volume_density = surface_density / z_guess
number_density = volume_density / c.me



def urad(T):
    return c.alpha * T**4

def ugas(n, T):
    gamma = 5/3
    return n*c.kb*T / (gamma - 1)

def effective_gamma(ugas, urad):
    gamma = 5/3
    return gamma - (gamma - 4/3) / (1 + ugas/urad)
    
def lerp(i):
    most = 1.55
    least = 1.43
    i /= 9
    return least + i * (most - least)
Ts = np.logspace(1, 14)
fig, axs = plt.subplots(3,3, figsize = (9,8), dpi = 300, tight_layout = True)
fig.suptitle('Shiokawa EoS', fontsize = 18)
axs = axs.flatten()
ns = np.logspace(-15, 2, num = 9) * number_density
for i, ax, n in zip(range(len(ns)), axs, ns):
    urads = [ urad(T)*n for T in Ts]
    ugass = [ ugas(n, T)*n for T in Ts]
    gammas = [ effective_gamma(gas, rad) for gas, rad in zip(ugass, urads) ]
    
    ax.set_title(f'$n_e$: {n:.2e} cm$-3$')
    ax.plot(Ts, gammas, '-o', c = 'k', 
              lw = 0.75, markersize = 1.5)
    
    ax2 = ax.twinx()
    ax2.plot(Ts, urads, '-o', c = c.AEK, 
              lw = 0.75, markersize = 1.5)
    ax2.plot(Ts, ugass, '-o', c = c.kroki, 
              lw = 0.75, markersize = 1.5)
    ax2.axhline(2e51, c = 'maroon', ls = '--')
    ax2.set_ylabel('Energy [erg]', color = 'darkorange')
    ax.set_ylabel('Effective $\Gamma$')
    ax.set_xlabel('Temperature [K]')
    
    ax2.spines['left'].set_color('k')
    ax.tick_params(axis='y', colors='k')
    ax2.spines['right'].set_color('darkorange')
    ax2.tick_params(axis='y', colors='darkorange')
    
    ax2.set_ylim(1e15, 1e65)
    ax2.set_yscale('log')
    ax.set_xscale('log')    
    
axs[3].text(1e2, 1.485, 'Gas', c=c.kroki, rotation = 12)
axs[3].text(1e2, 1.375, 'Radiation', c=c.AEK, rotation = 45)
    
    