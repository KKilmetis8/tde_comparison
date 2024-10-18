#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 13:09:14 2024

@author: konstantinos
"""

import gc
import csv
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import colorcet
import numba
from tqdm import tqdm

from src.ToyModel.solvers import regula_falsi
import src.Utilities.prelude as c
from src.Utilities.parser import parse
from src.Utilities.isalice import isalice
alice, plot = isalice()

#@numba.njit
def circ_locus(epsilon, Mbh):
    j = Mbh / np.sqrt(2*epsilon)
    return j

@numba.njit
def distance(e, ep, j, jp):
    return np.sqrt( (e-ep)**2 - (j-jp)**2)

@numba.njit
def d_prime(x, epsilon, j, Mbh):
    inv_sqrt2 = 1/np.sqrt(2)
    # oros0 = ((x-epsilon)**2 + (Mbh*inv_sqrt2*x**(-1/2) - j)**2)**(-1)/2
    oros1 = 2*(x-epsilon)
    par21 = Mbh*inv_sqrt2 * x**(-3/2)
    par22 = Mbh*inv_sqrt2 * x**(-1/2) - j
    return oros1 - par21*par22

@numba.njit
def d_primeprime(x, epsilon, j, Mbh):
    inv_sqrt2 = 1/np.sqrt(2)
    oros1 = 0.25 * Mbh**2 * x**(-3)
    par21 = Mbh * inv_sqrt2 * x**(-0.5) - j
    par22 = 0.75 * np.sqrt(2) * Mbh * x**(-2.5)
    return oros1 - par21*par22
    
def get_dist(energy, j, Mbh):
    e_closest = regula_falsi(a = 0.1*delta_e, b = 40*delta_e, f = d_prime, 
                          args = (energy, j, Mbh))
    if type(e_closest) == type(None):
        return np.NaN
    j_closest = circ_locus(e_closest, Mbh)
    dist = distance(e_closest, energy, j_closest, j)
    return dist

#%% Import data
if alice:
    pre = '/home/kilmetisk/data1/TDE/'
    args = parse()
    sim = args.name
    mstar = args.mass
    rstar = args.radius
    Mbh = args.blackhole
    fixes = [args.only]
    Mbh = float(Mbh)
    rg = 2*Mbh/c.c**2
    Rt = rstar * (Mbh/mstar)**(1/3)
    Rp = Rt
    jp = np.sqrt(2*Rp*Mbh)
    delta_e = mstar/rstar * ( (Mbh/mstar)**(1/3) - 1 )
    ecirc = Mbh/(4*Rt)
else:
    m = 5
    Mbh = 10**m
    pre = f'{m}/'
    fixes = [301, 302, 303]
    mstar = 0.5
    rstar = 0.47
    rg = 2*Mbh/c.c**2
    Rt = rstar * (Mbh/mstar)**(1/3)
    Rp = Rt
    jp = np.sqrt(2*Rp*Mbh)
    delta_e = mstar/rstar * ( (Mbh/mstar)**(1/3) - 1 )
    ecirc = Mbh/(4*Rt)
days = []
mean_dists = []
for idx_s, snap in enumerate(fixes):
    # Load data -----------------------------------------------------------------
    if alice:
        X = np.load(f'{pre}{sim}/snap_{snap}/CMx_{snap}.npy')
        Y = np.load(f'{pre}{sim}/snap_{snap}/CMy_{snap}.npy')
        Z = np.load(f'{pre}{sim}/snap_{snap}/CMz_{snap}.npy')
        VX = np.load(f'{pre}{sim}/snap_{snap}/Vx_{snap}.npy')
        VY = np.load(f'{pre}{sim}/snap_{snap}/Vy_{snap}.npy')
        VZ = np.load(f'{pre}{sim}/snap_{snap}/Vz_{snap}.npy')
        Den = np.load(f'{pre}{sim}/snap_{snap}/Den_{snap}.npy')
        Vol = np.load(f'{pre}{sim}/snap_{snap}/Vol_{snap}.npy')
        day = np.loadtxt(f'{pre}{sim}/snap_{snap}/tbytfb_{snap}.txt')
        days.append(day)
    else:
        X = np.load(f'{pre}{snap}/CMx_{snap}.npy')
        Y = np.load(f'{pre}{snap}/CMy_{snap}.npy')
        Z = np.load(f'{pre}{snap}/CMz_{snap}.npy')
        VX = np.load(f'{pre}{snap}/Vx_{snap}.npy')
        VY = np.load(f'{pre}{snap}/Vy_{snap}.npy')
        VZ = np.load(f'{pre}{snap}/Vz_{snap}.npy')
        Den = np.load(f'{pre}{snap}/Den_{snap}.npy')
        Vol = np.load(f'{pre}{snap}/Vol_{snap}.npy')

        day = np.loadtxt(f'{pre}{snap}/tbytfb_{snap}.txt')
        days.append(day)

    # Calculate energy & mask unbound, fluff
    R = np.sqrt(X**2 + Y**2 + Z**2)
    V = np.sqrt(VX**2 + VY**2 + VZ**2)
    Orb = 0.5*V**2 - Mbh / (R-rg)     
    boundmask = Orb < 0
    denmask = Den > 1e-10
    mask = boundmask * denmask
    Orb = np.abs(Orb[mask])
    
    # Calculate mass for weighing
    Mass = np.multiply(Den[mask], Vol[mask])
    inv_mass_sum = 1/np.sum(Mass)
#%%    
    # Calculate angular momentum
    jvec = np.cross( np.array([X[mask], Y[mask], Z[mask]]).T, 
                     np.array([VX[mask], VY[mask], VZ[mask]]).T
                   )
    j = np.linalg.norm(jvec, axis = 1)
    
    # Get distances
    dists = np.ones(len(Orb))
    for i in tqdm(range(len(Orb))):
        dists[i] = Mass[i] * get_dist(Orb[i], j[i], Mbh) * inv_mass_sum
    # Filter out the nans
    nanmask = ~np.isnan(dists)
    dists = dists[nanmask] 
    mean_dist = np.mean(dists)
    
    if alice:
        pre_saving = f'/home/kilmetisk/data1/TDE/tde_comparison/data/'
        filepath =  f'{pre_saving}tcirc/{sim}/meandists.csv'
        data = [snap, day, mean_dist]
        with open(filepath, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
        file.close()
#%% Distances plots
# fig, ax = plt.subplots(1,1, figsize = (4,3), dpi = 300)
# ax.plot(days, mean_dists, c='r', ls = '-')
# ax2 = ax.twinx()
# ax2.plot(days, t_circ, c='k', ls = '--')
# # ax2.set_ylim(0, 100)
# ax2.set_ylabel('$t_\mathrm{circ}$ [steps]')
# ax.set_xlabel('Time [$t_\mathrm{FB}$]')
# ax.set_ylabel('Distance to circ. locus')


    
    
    
