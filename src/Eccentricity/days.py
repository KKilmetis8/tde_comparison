#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 18:19:32 2023

@author: konstantinos
"""
# Vanilla
import numba
import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [8.0, 4.0]

# Chocolate
from src.Utilities.isalice import isalice
from src.Utilities.parser import parse
alice, plot = isalice()
pre = '/home/kilmetisk/data1/TDE/'

# Choose Simulation
if alice:
    args = parse()
    sim = args.name
    mstar = args.mass
    rstar = args.radius
    Mbh = args.blackhole
    Mbh = float(Mbh)
    Rt = rstar * (Mbh/mstar)**(1/3)
    fixes = np.arange(args.first, args.last + 1)
else:
    m = 10
    # do it yourself

save = True
method = 'caster'

# Constants
G = 6.6743e-11  # SI
Msol = 1.98847e30  # kg
Rsol = 6.957e8  # m

t = np.sqrt(Rsol**3 / (Msol*G))  # Follows from G=1
# Need these for the PW potential
c = 3e8 * t/Rsol  # c in simulator units.
rg = 2*Mbh/c**2
t_fall = 40 * (Mbh/1e6)**(0.5)  # days EMR+20 p13
apocenter = 0.5 * Rt * (Mbh/mstar)**(1/3)  # There is m_* hereeee
@numba.njit
def masker(arr, mask):
    len_bound = np.sum(mask)
    new_arr = np.zeros(len_bound)
    k = 0
    for i in range(len(arr)):
        if mask[i]:
            new_arr[k] = arr[i]
            k += 1
    return new_arr

# %%
# MAIN
colarr = []
fixdays = []
massarr = []
Tarr = []
smaarr = []
orbarr = []
jsqarr = []
for fix in fixes:

    if alice:
        t_by_tfb = np.loadtxt(f'{pre}{sim}/snap_{fix}/tbytfb_{fix}.txt')
        fixdays.append(t_by_tfb)
    else:
        day = np.round(days_since_distruption(
            sim + fix + '/snap_' + fix + '.h5'), 1)
        t_by_tfb = day  # /t_fall
        fixdays.append(t_by_tfb)

if save:
    if alice:
        print('saving...')
        #np.savetxt(f'{pre}tde_comparison/data/ecc{sim}.txt', colarr)
        np.savetxt(f'{pre}tde_comparison/data/ef8/eccdays{sim}.txt', fixdays)
        #np.savetxt(f'{pre}tde_comparison/data/eccT{sim}.txt', Tarr)
        #np.savetxt(f'{pre}tde_comparison/data/eccsemimajoraxis{sim}.txt', smaarr)
        np.savetxt(f'{pre}tde_comparison/data/ef8/eccenergy{sim}.txt', orbarr)
        # np.savetxt(f'{pre}tde_comparison/data/eccjsq{sim}.txt', jsqarr)
    else:
        with open('data/ecc'+ str(m) + '.txt', 'a') as file:
            for i in range(len(colarr)):
                file.write('# snap' + ' '.join(map(str, fixes[i])) + '\n')
                file.write('# Eccentricity \n') 
                file.write(' '.join(map(str, colarr[i])) + '\n')
            file.close() 
