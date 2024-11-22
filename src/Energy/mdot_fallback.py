#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 10:39:25 2024

@author: konstantinos
"""
import numpy as np
import matplotlib.pyplot as plt

import src.Utilities.prelude as c

def comeback(energy, Mbh):
    t = np.pi * Mbh / np.sqrt(2)
    t *= np.abs(energy)**(-3/2)
    return t
#%% Choose parameters ---
m = 4 # 4 5 6
Mbh = 10**m
pre = f'{m}/'
snap = 65 # # 65 80 145
mstar = 0.5
rstar = 0.47
deltaE = mstar/rstar * ((Mbh/mstar)**(1/3) + 1)
tfb = np.pi/np.sqrt(2) * np.sqrt(rstar**3/mstar * Mbh/mstar)

#%% Do it --- 
rg = 2*float(Mbh)/(c.c * c.t/c.Rsol_to_cm)**2
Rt = rstar * (Mbh/mstar)**(1/3) 

X = np.load(f'{pre}{snap}/CMx_{snap}.npy')
Y = np.load(f'{pre}{snap}/CMy_{snap}.npy')
Z = np.load(f'{pre}{snap}/CMz_{snap}.npy')
VX = np.load(f'{pre}{snap}/Vx_{snap}.npy')
VY = np.load(f'{pre}{snap}/Vy_{snap}.npy')
VZ = np.load(f'{pre}{snap}/Vz_{snap}.npy')
Den = np.load(f'{pre}{snap}/Den_{snap}.npy')
Vol = np.load(f'{pre}{snap}/Vol_{snap}.npy')
day = np.loadtxt(f'{pre}{snap}/tbytfb_{snap}.txt')
Parabolic_CM = np.genfromtxt(f'data/parabolic_orbit_{m}.csv', 
                     delimiter = ',')

# Frame of Ref
if snap<c.set_change(m):
    index = np.argmin(np.abs(day - Parabolic_CM.T[0]))
    X += Parabolic_CM.T[1][index]
    Y += Parabolic_CM.T[2][index]
    VX += Parabolic_CM.T[3][index]
    VY += Parabolic_CM.T[4][index]

# Calc
R = np.sqrt(X**2 + Y**2 + Z**2)
V = np.sqrt(VX**2 + VY**2 + VZ**2)
Orb = 0.5*V**2 - Mbh/(R - rg)
bound = Orb < 0
Mass = Vol * Den
Orb = Orb[bound]
fbtime = [ comeback(energy, Mbh) for energy in Orb]
fbtime = np.array(fbtime)
Mass = Mass[bound]
bins = np.logspace(-2.5, 4, 1000)


#%% Plot
elad_to_year = c.t * c.sec_to_yr 
plt.hist(fbtime * elad_to_year, color = 'k', bins = bins, weights = Mass)
plt.xscale('log')
plt.ylabel('Mass fallback rate $(\dot{M})$ [$M_\odot$/yr]')
plt.xlabel('Time [yr]')
plt.title(f'$10^{m}$ $M_\odot$, t = {day:.2f} t$_\mathrm{{fb}}$')

#%% Save
counts, bin_edges = np.histogram(fbtime, bins = bins, weights = Mass)
dMdt = np.zeros(len(counts))
for i in range(len(counts)):    
    dMdt[i] = counts[i]  / ( bin_edges[i+1] - bin_edges[i])
pre = 'data/tcirc/'
np.save(f'{pre}/mdot{m}', dMdt)
np.save(f'{pre}/time_for_mdot{m}', bins)
