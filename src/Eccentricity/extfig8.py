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
from src.Calculators.ONE_TREE_CASTER import BONSAI
from src.Calculators.casters import THE_SMALL_CASTER
from src.Extractors.time_extractor import days_since_distruption
from src.Eccentricity.eccentricity import e_calc, e_calc2
from src.Utilities.loaders import local_loader, alice_loader
from src.Utilities.isalice import isalice
from src.Utilities.parser import parse
alice, plot = isalice()

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
apocenter = 0.5 * Rt * (Mbh/mstar)**(1/3)  #

def masker(mask, list_of_quantities):
    new_list = []
    for quantity in list_of_quantities:
        new_list.append(quantity[mask])
    return (*new_list,)

# Do the thing
colarr = []
fixdays = []

for fix in fixes:
    if alice:
        fix = str(fix)
        pre = '/home/kilmetisk/data1/TDE/'
        # Import
        X, Y, Z, Vx, Vy, Vz, Den, Mass, day = alice_loader(sim, fix, 'orbital+den+mass',)
    else:
        X, Y, Z, Vx, Vy, Vz, Den, day = local_loader(m, fix, 'orbital+den')

    # Fluff mask
    denmask = np.where((Den > 1e-12))[0]
    X, Y, Z, Vx, Vy, Vz, Mass = masker(denmask, [X, Y, Z, Vx, Vy, Vz, Mass])
    # Make Bound Mask
    R = np.sqrt(np.power(X, 2) + np.power(Y, 2) + np.power(Z, 2))
    V = np.sqrt(np.power(Vx, 2) + np.power(Vy, 2) + np.power(Vz, 2))
    Orbital = (0.5 * V**2) - Mbh / (R-rg)
    bound_mask = np.where(Orbital < 0, 1, 0)

    # Apply Mask
    X, Y, Z, Vx, Vy, Vz, R, Mass = masker(bound_mask, [X, Y, Z, Vx, Vy, Vz, R, Mass])
    position = np.array((X, Y, Z)).T  # Transpose for col. vectors
    velocity = np.array((Vx, Vy, Vz)).T
    del X, Y, Z, Vx, Vy, Vz

    # EVOKE eccentricity
    # _, ecc, semi_major_axis = e_calc(position, velocity, Mbh)
    ecc = e_calc2(position, velocity, Mbh)
    # Cast down to 100 values
    radii = np.logspace(np.log10(0.4*Rt), np.log10(2*apocenter),
                        num=1000)  # simulator units

    if method == 'caster':
        print(fix)
        ecc_cast = THE_SMALL_CASTER(radii, R, ecc, weights = Mass)
        # T_cast = THE_SMALL_CASTER(radii, R_bound, T, weights=M)
        # semi_major_axis_cast = THE_SMALL_CASTER(radii, R_bound, 
        #                                       semi_major_axis, weights=M)
        # orbital_cast = THE_SMALL_CASTER(radii, R_bound, Orbital, weights = M)
        # jsq_cast = THE_SMALL_CASTER(radii, R_bound, jsq, weights=M)
    if method == 'tree':
        ecc_cast = BONSAI(radii, R, ecc)


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
        np.savetxt(f'{pre}tde_comparison/data/ecc2{sim}.txt', colarr)
        np.savetxt(f'{pre}tde_comparison/data/ef8/eccdays{sim}.txt', fixdays)

    else:
        with open('data/ecc'+ str(m) + '.txt', 'a') as file:
            for i in range(len(colarr)):
                file.write('# snap' + ' '.join(map(str, fixes[i])) + '\n')
                file.write('# Eccentricity \n') 
                file.write(' '.join(map(str, colarr[i])) + '\n')
            file.close() 
# %% Plotting
if plot:
    img = plt.pcolormesh(radii, fixdays, colarr,
                        cmap='jet', vmin=0, vmax=1)
    # plt.xlim(radii[0]-0.8 , radii[-12])
    cb = plt.colorbar(img)
    cb.set_label('Eccentricity')
    plt.xscale('log')
    plt.ylabel('Time [t/t$_{fb}$]', fontsize=14)
    plt.xlabel(r'Radius [$R_{\odot}$]', fontsize=14)
    plt.title('Eccentricity as a function of time and radius', fontsize=17)
    plt.show()