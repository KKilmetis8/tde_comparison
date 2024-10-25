#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 11:20:52 2024

@author: konstantinos
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from src.Utilities.isalice import isalice
import matplotlib.pyplot as plt
import csv

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
    opac_kind = 'LTE'
    m = int(np.log10(float(Mbh)))
else:
    m = 4
    Mbh = 10**m
    pre = f'{m}/'
    fixes = [50]
    mstar = 0.5
    rstar = 0.47
    deltaE = 2 * mstar/rstar * ((Mbh/mstar)**(1/3) + 1)

#%% Do it --- -----------------------------------------------------------------
days = []
rg = 2*Mbh/c.c**2
Rt = rstar * (Mbh/mstar)**(1/3) 
for idx_s, snap in enumerate(fixes):
    # Load data
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
        m = int(np.log10(float(Mbh)))
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
    step = 100
    plt.scatter(X[::step], Y[::step], c = np.log10(Den[::step]), cmap = 'cet_fire', s = 0.1)
    #%%
    R = np.sqrt(X**2 + Y**2 + Z**2)
    V = np.sqrt(VX**2 + VY**2 + VZ**2)
    Orb = 0.5*V**2 - Mbh/(R - rg) 
    Mass = np.multiply(Vol, Den)
    del X, Y, Z, VX, VY, VZ,
    
    plt.figure(figsize = (3,3), dpi = 300)
    plt.hist(Orb/deltaE, color='k', edgecolor = 'white', bins = 20, 
             range = (-1000, 1000), density = False, weights = Mass,)
    # plt.xlim(-10, 10)
    plt.text(350, 1e-1, f'Min: {np.min(Orb/deltaE):.0f}', 
             bbox = dict(boxstyle='square', facecolor='white', alpha=0.5))
    plt.yscale('log')
    plt.xlabel('Orbital Energy $[\Delta E_\mathrm{min}]$')
    plt.ylabel('Mass weighted Counts')
    plt.ylim(1e-9, 1)
    plt.title(f'$10^{m} M_\odot$ $|$ {day:.3f} $t_\mathrm{{FB}}$')

    if alice:
        # Save plot
        figsave = f'/home/kilmetisk/data1/TDE/figs/ehist/'
        plt.savefig(f'{figsave}{m}ehist{snap}.png')
        plt.close()
        
        # Save min
        pre_saving = f'/home/kilmetisk/data1/TDE/tde_comparison/data/'
        filepath =  f'{pre_saving}/energyhist/orbmins.csv'
        data = [snap, day, np.min(Orb/deltaE)]
        with open(filepath, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
        file.close()
