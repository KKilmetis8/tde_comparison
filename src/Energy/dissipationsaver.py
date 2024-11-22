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
def E_arrive(t, Mbh):
    '''Calculates what the energy neccecery is to come back
    at a certain time. Assumming Keplerian orbits.'''
    E = 0.5 * np.pi**2 * Mbh**2 / t**2    
    return -E**(1/3)
#%% Choose parameters -----------------------------------------------------------------
if alice:
    pre = '/home/kilmetisk/data1/TDE/'
    args = parse()
    sim = args.name
    mstar = args.mass
    rstar = args.radius
    Mbh = args.blackhole
    tfb = np.pi/np.sqrt(2) * np.sqrt(rstar**3/mstar * float(Mbh)/mstar)

    m = int(np.log10(float(Mbh)))
    fixes = np.arange(args.first, args.last + 1)
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
    fixes = [348] #[301, 302, 303]
    mstar = 0.5
    rstar = 0.47

#%% Opacities -----------------------------------------------------------------
days = []
rg = 2*float(Mbh)/(c.c * c.t/c.Rsol_to_cm)**2
for idx_s, snap in enumerate(fixes):
    print('Snapshot: ', snap)
    # Load data -----------------------------------------------------------------
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
        # T = np.load(f'{pre}{sim}/snap_{snap}/T_{snap}.npy')
        Diss = np.load(f'{pre}{sim}/snap_{snap}/Diss_{snap}.npy')
        # Rad = np.load(f'{pre}{sim}/snap_{snap}/Rad_{snap}.npy')
        Den = np.load(f'{pre}{sim}/snap_{snap}/Den_{snap}.npy')
        Vol = np.load(f'{pre}{sim}/snap_{snap}/Vol_{snap}.npy')
        day = np.loadtxt(f'{pre}{sim}/snap_{snap}/tbytfb_{snap}.txt')
        Parabolic_CM = np.genfromtxt(f'{pre}/tde_comparison/data/parabolic_orbit_{m}.csv'
                                , delimiter = ',')
        days.append(day)
    else:
        X = np.load(f'{pre}{snap}/CMx_{snap}.npy')
        Y = np.load(f'{pre}{snap}/CMy_{snap}.npy')
        Z = np.load(f'{pre}{snap}/CMz_{snap}.npy')
        VX = np.load(f'{pre}{snap}/Vx_{snap}.npy')
        VY = np.load(f'{pre}{snap}/Vy_{snap}.npy')
        VZ = np.load(f'{pre}{snap}/Vz_{snap}.npy')
        # T = np.load(f'{pre}{snap}/T_{snap}.npy')
        Diss = np.load(f'{pre}{snap}/Diss_{snap}.npy')
        Den = np.load(f'{pre}{snap}/Den_{snap}.npy')
        # IE = np.load(f'{pre}{snap}/IE_{snap}.npy')
        # Vol = np.load(f'{pre}{snap}/Vol_{snap}.npy')
        day = np.loadtxt(f'{pre}{snap}/tbytfb_{snap}.txt')
        days.append(day)

    if snap<change:
        index = np.argmin(np.abs(day - Parabolic_CM.T[0]))
        X += Parabolic_CM.T[1][index]
        Y += Parabolic_CM.T[2][index]
        VX += Parabolic_CM.T[3][index]
        VY += Parabolic_CM.T[4][index]

    R = np.sqrt(X**2 + Y**2 + Z**2)
    V = np.sqrt(VX**2 + VY**2 + VZ**2)
    Mass = np.multiply(Den, Vol)
    Earr = E_arrive(day * tfb, float(Mbh))
    Orb = 0.5*V**2 - Mbh/(R - rg) 
    hascomeback_mask = Orb < Earr
    Orb *= Mass
    bound_mask = Orb < 0
    del X, Y, Z, VX, VY, VZ, Den, R, V
    Diss *= Vol
    mask = bound_mask * hascomeback_mask
    sum_diss_masked = np.sum(Diss[mask])
    sum_diss = np.sum(Diss[bound_mask])
    
    if alice:
        pre_saving = f'/home/kilmetisk/data1/TDE/tde_comparison/data/'
        filepath =  f'{pre_saving}/energyhist/sum{m}diss.csv'
        data = [day, sum_diss, sum_diss_masked]
        with open(filepath, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
        file.close()


