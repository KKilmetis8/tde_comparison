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
    fixes = [199]
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
for idx_s, snap in enumerate(fixes):
    # Load data
    if alice:
        X = np.load(f'{pre}{sim}/snap_{snap}/CMx_{snap}.npy')
        Y = np.load(f'{pre}{sim}/snap_{snap}/CMy_{snap}.npy')
        Z = np.load(f'{pre}{sim}/snap_{snap}/CMz_{snap}.npy')
        Den = np.load(f'{pre}{sim}/snap_{snap}/Den_{snap}.npy')
        Vol = np.load(f'{pre}{sim}/snap_{snap}/Vol_{snap}.npy')
        day = np.loadtxt(f'{pre}{sim}/snap_{snap}/tbytfb_{snap}.txt')
        m = int(np.log10(float(Mbh)))
    else:
        X = np.load(f'{pre}{snap}/CMx_{snap}.npy')
        Y = np.load(f'{pre}{snap}/CMy_{snap}.npy')
        Z = np.load(f'{pre}{snap}/CMz_{snap}.npy')
        Den = np.load(f'{pre}{snap}/Den_{snap}.npy')
        Vol = np.load(f'{pre}{snap}/Vol_{snap}.npy')
        day = np.loadtxt(f'{pre}{snap}/tbytfb_{snap}.txt')
        

    # Mask maker 
    xlim = 10
    xmask_m = X > (Rt - xlim/2) 
    xmask_p = X < (Rt + xlim/2)
    xmask = xmask_m * xmask_p
    
    ylim = 2
    ymask_m = Y < (ylim/2) 
    ymask_p = Y > (-ylim/2)
    ymask = ymask_m * ymask_p
    
    zlim = 5
    zmask_m = Z < (zlim/2) 
    zmask_p = Z > (-zlim/2)
    zmask = zmask_m * zmask_p
    
    boxmask = xmask * ymask * zmask
    
    # Use it    
    Mass = Den * Vol
    mass_in_box = np.sum(Mass[boxmask])

    # Save
    if alice:
        pre_saving = f'/home/kilmetisk/data1/TDE/tde_comparison/data/'
        filepath =  f'{pre_saving}/energyhist/massinperi{m}.csv'
        data = [snap, day, mass_in_box]
        with open(filepath, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
        file.close()
    
    
    