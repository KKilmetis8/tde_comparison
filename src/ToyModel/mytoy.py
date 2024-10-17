#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:55:34 2024

@author: konstantinos
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import numba

from src.ToyModel.solvers import regula_falsi
import src.Utilities.prelude as c
pre = 'data/'
rstar = 0.47
mstar = 0.5
Mbh = 1e5
Rt = rstar * (Mbh/mstar)**(1/3)
Rp = Rt
jp = np.sqrt(2*Rp*Mbh)
delta_e = mstar/rstar * ( (Mbh/mstar)**(1/3) - 1 )
ecirc = Mbh/(4*Rt)

#@numba.njit
def circ_locus(epsilon, Mbh):
    j = Mbh / np.sqrt(2*epsilon)
    return j

@numba.njit
def distance(e, ep, j, jp):
    return np.sqrt( (e-ep)**2 - (j-jp)**2)

@numba.njit
def d_prime(x, epsilon, j, Mbh):
    inv_sqrt2 = 1/np.sqrt(2)
    # oros0 = ((x-epsilon)**2 + (Mbh*inv_sqrt2*x**(-1/2) - j)**2)**(-1)/2
    oros1 = 2*(x-epsilon)
    par21 = Mbh*inv_sqrt2 * x**(-3/2)
    par22 = Mbh*inv_sqrt2 * x**(-1/2) - j
    return oros1 - par21*par22

@numba.njit
def d_primeprime(x, epsilon, j, Mbh):
    inv_sqrt2 = 1/np.sqrt(2)
    oros1 = 0.25 * Mbh**2 * x**(-3)
    par21 = Mbh * inv_sqrt2 * x**(-0.5) - j
    par22 = 0.75 * np.sqrt(2) * Mbh * x**(-2.5)
    return oros1 - par21*par22

def baseplot(Mbh, ecirc, delta_energy_max):
    theory_es = np.linspace(0.1*delta_energy_max, 100*delta_energy_max, 100)
    theory_js = [ circ_locus(e, Mbh) for e in theory_es]
    fig, ax = plt.subplots(figsize = (3,3))
    plt.plot(theory_es/delta_e, theory_js/jp, color = 'k', alpha = 0.5)
    
    # Lines
    plt.axhline(jp/jp, color = 'k', ls = '--')
    plt.axvline(ecirc/delta_energy_max, color = 'k', ls='--')
    
    # Scales
    plt.xlim(0, 20) #ecirc/delta_energy_max*100 )
    plt.ylim(0.5,1.5) #plt.ylim(1e-1, 1e1)
    plt.xlabel('Orbital Energy $[\Delta E_\mathrm{max}]$')
    plt.ylabel('Angular Momentum $[j_\mathrm{par}]$')
    # plt.xscale('log')
    # plt.yscale('log')

    # Text
    # plt.text(0.16, 0.17, f'{day:.2f} $t_\mathrm{{FB}}$', fontsize = 13, 
    #          transform = fig.transFigure,
    #          bbox=dict(facecolor='whitesmoke', edgecolor='black'))
    plt.text(12, 0.6, r'$E_\mathrm{circ}$', fontsize = 10)
    plt.text(0.5, 0.80, r'$j = M_\mathrm{BH} \epsilon^{-1/2}$', fontsize = 10, 
             transform = fig.transFigure,
             bbox=dict(facecolor='whitesmoke', edgecolor='black'))
    # cb.set_label('Eccentricity')
    m = int(np.log10(Mbh))
    plt.title(f'Action Space $|$ $10^{m} M_\odot$')

class Toy:
    def __init__(self, energy, j, mass, mass_tot, Mbh):
        self.energy = energy
        self.j = j
        self.mass = mass
        self.mass_tot = mass_tot
        self.Mbh = Mbh
        
        # Init lists
        self.dist_hist = []
        e_closest = regula_falsi(a = 0.1*delta_e, b = 40*delta_e, f = d_prime, 
                              args = (self.energy, self.j, self.Mbh))
        j_closest = circ_locus(e_closest, self.Mbh)
        dist = distance(e_closest, self.energy, j_closest, self.j)
        self.dist_hist.append( self.mass * dist / self.mass_tot )
        self.energy_hist = [energy/delta_e]
        self.j_hist = [j/jp]
    
    def get_dist(self):
        if self.dist_hist[-1] < 0.2:
            self.dist_hist.append(self.dist_hist[-1])
            return 
        e_closest = regula_falsi(a = 0.1*delta_e, b = 40*delta_e, f = d_prime, 
                              args = (self.energy, self.j, self.Mbh))
        if type(e_closest) == type(None):
            self.dist_hist.append(self.dist_hist[-1])
            return
        j_closest = circ_locus(e_closest, self.Mbh)
        dist = distance(e_closest, self.energy, j_closest, self.j)
        self.dist_hist.append( self.mass * dist / self.mass_tot )
        return dist
    
    def change_E(self, dE):
        if self.dist_hist[-1] < 0.2:
            return 
        self.energy_hist.append(self.energy/delta_e)
        self.energy += dE
        
    def change_j(self, dj):
        if self.dist_hist[-1] < 0.2:
            return 
        self.j_hist.append(self.j/jp)
        self.j += dj
#%%
# Make some toys
ntoys = 100
mass_sum = 0.25
masses = np.random.rand(ntoys)
masses = masses / np.sum(masses) * mass_sum
energies = np.random.rand(ntoys) * delta_e
toys = []

for i in range(ntoys):
    toys.append( Toy(energies[i], jp, mass = masses[i], 
                     mass_tot = mass_sum, Mbh = Mbh))

# Evolve the system
steps = 85
dE = 0.2 * delta_e
for i in range(steps):
    j_changes = np.random.rand(ntoys//2)
    for j in range(len(toys)//2):
        dist = toys[j].get_dist()
        dj = 1e-2 * np.random.rand() * jp * np.random.choice((-1,1))
        if type(dist) == type(None):
            continue
        if dist<0.1:
            continue
        toys[j].change_E(dE)
        toys[j].change_j(dj)
        j_changes[j] = dj
        
    # Enforce global j-conservation
    for j in range(ntoys//2, ntoys):
        dist = toys[j].get_dist()
        if type(dist) == type(None):
            continue
        if dist<0.1:
            continue
        toys[j].change_E(dE)
        toys[j].change_j(-j_changes[j//2])
#%% Calculate t_circ
dists = np.zeros(steps+1) # initial
for i in range(ntoys):
    dists_this = np.array(toys[i].dist_hist)
    dists += dists_this
dists /= ntoys
dists_dot = np.gradient(dists)
nonzeromask = dists_dot != 0
t_circ = np.abs( (dists[nonzeromask] - 0.2)/dists_dot[nonzeromask])

#%% Distances plots
fig, ax = plt.subplots(1,1, figsize = (4,3), dpi = 300)
step = 1
colors = [c.c91, c.c92, c.c93, c.c94, c.c95, c.c96, c.c97, c.c98, c.c99]
for i, col in enumerate(colors):
    plt.plot(toys[i*2].dist_hist, 
            '-', c = col,  alpha = 50*toys[i*2].mass/mass_sum)
plt.plot(dists, c='r', ls = '-')
ax2 = ax.twinx()
ax2.plot(t_circ, c='k', ls = '--')
ax2.set_ylim(0, 100)
ax2.set_ylabel('$t_\mathrm{circ}$ [steps]')
ax.set_xlabel('Time [steps]')
ax.set_ylabel('Distance to circ. locus')
#%%
baseplot(Mbh, ecirc, delta_e)
step = 1
for i, col in enumerate(colors):
    plt.plot(toys[i*2].energy_hist[0], toys[i*2].j_hist[0], 
             'h', c = col, markersize = 4)
    plt.plot(toys[i*2].energy_hist[::step], toys[i*2].j_hist[::step], 
            '-', c = col, markersize = 2, alpha = 0.5)
    plt.plot(toys[i*2].energy_hist[-1], toys[i*2].j_hist[-1], 
            'h', c = col, markersize = 4)
# plt.plot([toy1.energy/delta_e, ec/delta_e], [toy1.j/jp, jc/jp], ':h', c='k', lw=1,
#           markersize = 3)
# plt.plot(toy1.energy/delta_e, toy1.j/jp, 'h', c='b', markersize = 4)
# plt.plot(ec/delta_e, jc/jp, 'h', c='r', markersize = 4)

