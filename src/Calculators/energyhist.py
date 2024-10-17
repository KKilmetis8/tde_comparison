#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 19:50:13 2024

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
    # fixes = np.arange(args.first, args.last + 1)
    opac_kind = 'LTE'
    m = 'AEK'
    check = 'MONO AEK'

    if Mbh == 10_000:
        if 'HiRes' in sim:
            fixes = [210]
            print('BH 4 HiRes')

        else:
            fixes = np.arange(80,348+1) # [164, 237, 313]
            print('BH 4')
    elif Mbh == 100_000:
        fixes = np.arange(132,365+1) # [208, 268,]# 365]
        print('BH 5')
    else:
        Mbh = 1e6
        fixes = np.arange(180,414+1)
        print('BH 6')
else:
    m = 4
    Mbh = 10**m
    pre = f'{m}/'
    fixes = [348] #[301, 302, 303]
    mstar = 0.5
    rstar = 0.47

#%% Opacities -----------------------------------------------------------------
days = []
rg = 2*Mbh/c.c**2
Rt = rstar * (Mbh/mstar)**(1/3) 
for idx_s, snap in enumerate(fixes):
    print('Snapshot: ', snap)
    # Load data -----------------------------------------------------------------
    if alice:
        X = np.load(f'{pre}{sim}/snap_{snap}/CMx_{snap}.npy')
        Y = np.load(f'{pre}{sim}/snap_{snap}/CMy_{snap}.npy')
        Z = np.load(f'{pre}{sim}/snap_{snap}/CMz_{snap}.npy')
        VX = np.load(f'{pre}{sim}/snap_{snap}/Vx_{snap}.npy')
        VY = np.load(f'{pre}{sim}/snap_{snap}/Vy_{snap}.npy')
        VZ = np.load(f'{pre}{sim}/snap_{snap}/Vz_{snap}.npy')
        # T = np.load(f'{pre}{sim}/snap_{snap}/T_{snap}.npy')
        Den = np.load(f'{pre}{sim}/snap_{snap}/Den_{snap}.npy')
        Rad = np.load(f'{pre}{sim}/snap_{snap}/Rad_{snap}.npy')
        IE = np.load(f'{pre}{sim}/snap_{snap}/IE_{snap}.npy')
        Vol = np.load(f'{pre}{sim}/snap_{snap}/Vol_{snap}.npy')
        day = np.loadtxt(f'{pre}{sim}/snap_{snap}/tbytfb_{snap}.txt')
        box = np.load(f'{pre}{sim}/snap_{snap}/box_{snap}.npy')
        days.append(day)
    else:
        X = np.load(f'{pre}{snap}/CMx_{snap}.npy')
        Y = np.load(f'{pre}{snap}/CMy_{snap}.npy')
        Z = np.load(f'{pre}{snap}/CMz_{snap}.npy')
        VX = np.load(f'{pre}{snap}/Vx_{snap}.npy')
        VY = np.load(f'{pre}{snap}/Vy_{snap}.npy')
        VZ = np.load(f'{pre}{snap}/Vz_{snap}.npy')
        # T = np.load(f'{pre}{snap}/T_{snap}.npy')
        Den = np.load(f'{pre}{snap}/Den_{snap}.npy')
        Rad = np.load(f'{pre}{snap}/Rad_{snap}.npy')
        IE = np.load(f'{pre}{snap}/IE_{snap}.npy')
        Vol = np.load(f'{pre}{snap}/Vol_{snap}.npy')
        day = np.loadtxt(f'{pre}{snap}/tbytfb_{snap}.txt')
        box = np.load(f'{pre}{snap}/box_{snap}.npy')
        days.append(day)

    R = np.sqrt(X**2 + Y**2 + Z**2)
    V = np.sqrt(VX**2 + VY**2 + VZ**2)
    Mass = np.multiply(Den, Vol)
    # Erad = np.multiply(Rad, Mass)
    # Eie = np.multiply(IE, Mass)
    denmask = Den > 1e-19
    rmask_in = R > 0.6 * Rt
    rmask_out = R < 5 * Rt
    mask = denmask * rmask_in * rmask_out
    Orb = 0.5*V**2 - Mbh/(R - rg) 

    del X, Y, Z, VX, VY, VZ, Mass, Vol, Den, R, V
    
    mean_orb = np.mean(np.abs(Orb[mask]))
    mean_rad = np.mean(Rad[mask])
    mean_IE = np.mean(IE[mask])
    
    if alice:
        pre_saving = f'/home/kilmetisk/data1/TDE/tde_comparison/data'
        filepath =  f'{pre_saving}/energyhist/{sim}/ehist_spec.csv'
        data = [day, mean_orb, mean_IE, mean_rad]
        with open(filepath, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
        file.close()