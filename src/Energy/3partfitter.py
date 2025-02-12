#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 21:44:36 2024

@author: konstantinos
"""

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
from src.Utilities.parser import parse

# Constants
m = 4
Mbh = 10**m
pre = f'{m}/'
if m == 4:
    snap = 65 # 65 80 145
if m == 5:
    snap = 80
if m == 6:
    snap = 145
mstar = 0.5
rstar = 0.47
deltaE = mstar/rstar * ((Mbh/mstar)**(1/3) + 1)
rg = 2*float(Mbh)/(c.c * c.t/c.Rsol_to_cm)**2
Rt = rstar * (Mbh/mstar)**(1/3) 
if m == 4:
    change = 80
if m == 5:
    change = 132
if m == 6:
    change = 180

# Load data
X = np.load(f'{pre}{snap}/CMx_{snap}.npy')
Y = np.load(f'{pre}{snap}/CMy_{snap}.npy')
Z = np.load(f'{pre}{snap}/CMz_{snap}.npy')
VX = np.load(f'{pre}{snap}/Vx_{snap}.npy')
VY = np.load(f'{pre}{snap}/Vy_{snap}.npy')
VZ = np.load(f'{pre}{snap}/Vz_{snap}.npy')
Den = np.load(f'{pre}{snap}/Den_{snap}.npy')
Vol = np.load(f'{pre}{snap}/Vol_{snap}.npy')
day = np.loadtxt(f'{pre}{snap}/tbytfb_{snap}.txt')
#
m = int(np.log10(float(Mbh)))
Parabolic_CM = np.genfromtxt(f'data/parabolic_orbit_{m}.csv', 
                             delimiter = ',')

# Plot
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
# Orb *= Mass
del X, Y, Z, VX, VY, VZ,
plt.hist(Orb)

# Fitting
bin_num = 10000
counts, bin_edges = np.histogram(Orb, bin_num, range = (-5*deltaE, 5*deltaE), 
                            weights = Mass)
dMdE = counts  / ( bin_edges[1] - bin_edges[0])
bin_centers = np.array([0.5*(bin_edges[i+1] + bin_edges[i]) for i in range(len(bin_edges)-1)])
# x_cutoff = 0.9  # Define a cutoff for the flat-top region
# center_mask = np.abs(bin_centers) < x_cutoff  # Center region mask
# wing_mask = np.abs(bin_centers) >= x_cutoff  # Wing regions mask

# # Top is Rossi+19 eq/23
norm = mstar / (2 * deltaE)
# flat_top_inter = np.interp(bin_centers[center_mask], 
#                            bin_centers, dMdE)
# # flat_top_value = np.mean(dMdE[center_mask]) * norm
# # flat_top_interp = np.full(sum(center_mask), flat_top_value)

# # Interpolate the wings using linear interpolation on each side
# left_wing_inter = np.interp(bin_centers[wing_mask & (bin_centers < 0)], 
#                              bin_centers, dMdE)
# right_wing_inter = np.interp(bin_centers[wing_mask & (bin_centers > 0)], 
#                               bin_centers, dMdE)

# # Combine interpolated parts
# fitted_y = np.empty_like(dMdE)
# fitted_y[center_mask] = flat_top_inter
# fitted_y[wing_mask & (bin_centers < 0)] = left_wing_inter
# fitted_y[wing_mask & (bin_centers > 0)] = right_wing_inter
#
fitted_y = np.interp(bin_centers, bin_centers, dMdE) 

fig, axs = plt.subplots(1, 1, figsize = (3,3), dpi = 300,)# tight_layout = True)
axs.hist(Orb/deltaE, color='k', bins = bin_num, weights = Mass,
         range = (-5, 5))
axs.axvline(1, c = c.AEK, ls ='--')
axs.axvline(-1, c = c.AEK, ls ='--')
axs.axvline(0, c = 'white', ls =':')
Ecirc = Mbh/(4*Rt)
# axs.axvline(-Ecirc/deltaE, c = 'g', ls ='-')

# axs.plot(bin_centers/deltaE, dMdE, 'b')
axs.plot(bin_centers/deltaE, fitted_y, 'r', ls = ':')


axs.axhline(norm, c = 'k', ls = ':')
# Make pretty 
axs.set_yscale('log')
axs.set_xlabel('Orbital Energy $[\Delta E_\mathrm{min}]$')
axs.set_ylabel('dM/dE')
# axs.text(2.5, norm*2, '$M_*/2\Delta E$')
fig.suptitle(f'$10^{m} M_\odot$ $|$ {day:.3f} $t_\mathrm{{FB}}$', y = 0.97)

# Save
np.save(f'data/tcirc/m_calli{m}', fitted_y)
np.save(f'data/tcirc/e_calli{m}', bin_centers)