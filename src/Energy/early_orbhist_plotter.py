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
plt.rcParams['text.usetex'] = False

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
    m = int(np.log10(float(Mbh)))
    deltaE = mstar/rstar * ((float(Mbh)/mstar)**(1/3) + 1)
    if m == 4:
        change = 80
    if m == 5:
        change = 131
    if m == 6:
        change = 180
else:
    m = 4
    Mbh = 10**m
    pre = f'{m}/'
    fixes = [50]
    mstar = 0.5
    rstar = 0.47
    deltaE = mstar/rstar * ((Mbh/mstar)**(1/3) + 1)

#%% Do it --- -----------------------------------------------------------------
rg = 2*float(Mbh)/(c.c * c.t/c.Rsol_to_cm)**2
Rt = rstar * (Mbh/mstar)**(1/3) 
for idx_s, snap in enumerate(fixes):
    # Load data
    if alice:
        try:
            X = np.load(f'{pre}{sim}/snap_{snap}/CMx_{snap}.npy')
        except:
            continue
        Y = np.load(f'{pre}{sim}/snap_{snap}/CMy_{snap}.npy')
        Z = np.load(f'{pre}{sim}/snap_{snap}/CMz_{snap}.npy')
        VX = np.load(f'{pre}{sim}/snap_{snap}/Vx_{snap}.npy')
        VY = np.load(f'{pre}{sim}/snap_{snap}/Vy_{snap}.npy')
        VZ = np.load(f'{pre}{sim}/snap_{snap}/Vz_{snap}.npy')
        Den = np.load(f'{pre}{sim}/snap_{snap}/Den_{snap}.npy')
        Vol = np.load(f'{pre}{sim}/snap_{snap}/Vol_{snap}.npy')
        day = np.loadtxt(f'{pre}{sim}/snap_{snap}/tbytfb_{snap}.txt')
        m = int(np.log10(float(Mbh)))
        Parabolic_CM = np.genfromtxt(f'{pre}/tde_comparison/data/parabolic_orbit_{m}.csv'
                                     , delimiter = ',')
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
        m = int(np.log10(float(Mbh)))
        Parabolic_CM = np.genfromtxt(f'data/parabolic_orbit_{m}.csv', 
                                     delimiter = ',')

    #
    if snap<change:
        index = np.argmin(np.abs(day - Parabolic_CM.T[0]))
        X += Parabolic_CM.T[1][index]
        Y += Parabolic_CM.T[2][index]
        VX += Parabolic_CM.T[3][index]
        VY += Parabolic_CM.T[4][index]

    R = np.sqrt(X**2 + Y**2 + Z**2)
    V = np.sqrt(VX**2 + VY**2 + VZ**2)
    Orb = 0.5*V**2 - Mbh/(R - rg) 
    Mass = np.multiply(Vol, Den)
    del X, Y, Z, VX, VY, VZ,
    
    fig, axs = plt.subplots(1, 1, figsize = (4,4), dpi = 300,)# tight_layout = True)
    axs.hist(Orb/deltaE, color='k', bins = 100, 
             range = (-5, 5), weights = Mass,)
    axs.axvline(1, c = c.AEK, ls ='--')
    axs.axvline(-1, c = c.AEK, ls ='--')
    axs.axvline(0, c = 'white', ls =':')

    axs.set_ylim(1e-10, 1e-1)

    # axs[1].hist(Orb/deltaE, color='k', bins = 1000, 
    #          range = (-1000, 1000),  weights = Mass,)
    # axs[1].text(100, 1e-2, f'Min: {np.min(Orb/deltaE):.0f}', 
    #          bbox = dict(boxstyle='square', facecolor='white', alpha=0.5))
    # axs[1].set_ylim(1e-15, 2)

    # Make pretty 
    # axs[0].set_yscale('log')
    axs.set_yscale('log')
    axs.set_xlabel('Orbital Energy [DeltaE]')
    axs.set_ylabel('Mass weighted Counts')
    axs.set_title(f'logMbh : {m}  - time: {day:.3f} t_fb')

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

