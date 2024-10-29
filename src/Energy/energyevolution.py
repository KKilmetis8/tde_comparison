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
save = False
if alice:
    pre = '/home/kilmetisk/data1/TDE/'
    args = parse()
    sim = args.name
    mstar = args.mass
    rstar = args.radius
    Mbh = args.blackhole
    fixes = np.arange(args.first, args.last + 1)
    opac_kind = 'LTE'
    m = 'AEK'
    check = 'MONO AEK'

    if Mbh == 10_000:
        if 'HiRes' in sim:
            fixes = [210]
            print('BH 4 HiRes')

        else:
            fixes = [272] # [164, 237, 313]
            print('BH 4')
    elif Mbh == 100_000:
        fixes = [302] # [208, 268,]# 365]
        print('BH 5')
    else:
        Mbh = 1e6
        fixes = [351]
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
        del day
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
    Erad = np.multiply(Rad, Mass)
    Eie = np.multiply(IE, Mass)
    denmask = Den > 1e-19
    rmask_in = R > 0.6 * Rt
    rmask_out = R < 3 * Rt
    mask = denmask * rmask_in * rmask_out
    Orb = 0.5*Mass*V**2 - Mass*Mbh/(R - rg) 

    del X, Y, Z, VX, VY, VZ, Mass, Vol, Den, R, V
    
    mean_orb = np.mean(np.abs(Orb[mask]))
    mean_rad = np.mean(Erad[mask])
    mean_IE = np.mean(Eie[mask])
    
    if alice:
        pre_saving = f'/home/s3745597/data1/TDE/tde_comparison/data/'
        filepath =  f'{pre_saving}/energyhist/{sim}/eladred_d19_many.csv'
        data = [day, mean_orb, mean_IE, mean_rad]
        with open(filepath, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
        file.close()
#%%
# plt.figure(figsize = (3,3))
# plt.bar(days[0], mean_orb, color='k')
# plt.bar(days[0], mean_IE, color='tab:orange', bottom = mean_orb)
# plt.bar(days[0], mean_rad, color=c.AEK, bottom = mean_orb + mean_IE)
