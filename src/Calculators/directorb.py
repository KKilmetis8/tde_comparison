#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 18:32:11 2024

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

#%% Import data
if alice:
    pre = '/home/kilmetisk/data1/TDE/'
    args = parse()
    sim = args.name
    mstar = args.mass
    rstar = args.radius
    Mbh = args.blackhole
    fixes = np.arange(args.first, args.last + 1)
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
    print(snap)
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
    # Calc. R, V
    R = np.sqrt(X**2 + Y**2 + Z**2)
    del X, Y, Z
    gc.collect()
    V = np.sqrt(VX**2 + VY**2 + VZ**2)
    del VX, VY, VZ
    gc.collect()
    
    # Calc. energy & mask unbound, fluff
    Orb = 0.5*V**2 - Mbh / (R-rg)     
    boundmask = Orb < 0
    denmask = Den > 1e-18
    mask = boundmask * denmask
    Orb = np.abs(Orb[mask])
    
    # Calc. mass for weighing
    Mass = np.multiply(Den[mask], Vol[mask])
    inv_mass_sum = 1/np.sum(Mass)
    
    # Orb - ecirc
    orb_minus_ecirc = np.add(Orb, -ecirc)
    mw_orb_minus_ecirc = np.multiply(Mass, orb_minus_ecirc) * inv_mass_sum
    mw_mean_orb_minus_ecirc = np.mean(mw_orb_minus_ecirc)
    mean_orb_minus_ecirc = np.mean(orb_minus_ecirc)
    
    # Just orb
    mw_orb = np.multiply(Mass, Orb) * inv_mass_sum
    mw_mean_orb = np.mean(mw_orb)
    mean_orb = np.mean(Orb)
    
    # Save data 
    if alice:
        pre_saving = '/home/kilmetisk/data1/TDE/tde_comparison/data/'
        m = int(np.log10(float(Mbh)))
        filepath =  f'{pre_saving}tcirc/tcircdirect{m}.csv'
        data = [snap, day, mean_orb, mean_orb_minus_ecirc, mw_mean_orb, mw_mean_orb_minus_ecirc]
        with open(filepath, 'a', newline='') as file:
            # file.write('# snap, time [tfb], mean orb, mean orb - Ecirc, mean mass weighted orb, mean mass weighted orb - Ecirc \n')
            writer = csv.writer(file)
            writer.writerow(data)
        file.close()
    