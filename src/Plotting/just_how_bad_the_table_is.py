#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 19:22:56 2023

@author: konstantinos
"""
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.facecolor']= 	'whitesmoke'
import colorcet
from astropy.coordinates import cartesian_to_spherical
from src.Calculators.legion_of_casters import THROUPLE_S_CASTERS

#%% Constants & Converter
NSIDE = 4
G = 6.6743e-11 # SI
Msol = 1.98847e30 # kg
Rsol = 6.957e8 # m
t = np.sqrt(Rsol**3 / (Msol*G )) # Follows from G = 1
c = 3e8 * t/Rsol # simulator units. Need these for the PW potential
c_cgs = 3e10 # [cm/s]
Msol_to_g = 1.989e33 # [g]
Rsol_to_cm = 6.957e10 # [cm]
den_converter = Msol_to_g / Rsol_to_cm**3
en_den_converter = Msol_to_g / (Rsol_to_cm  * t**2 ) # Energy Density converter

#%%

# ax.plot(np.log10(Den_triple_casted.ravel() ), np.log10(T_triple_casted.ravel()), 
#         'x', c='k', markersize = 1)
# ax.plot([],[], 'x', c = 'k', label = 'Triple' )
fix = '844'
m = 6
fix = str(fix)
Mbh = 10**m 
Rt =  Mbh**(1/3) # Msol = 1, Rsol = 1

# Import
X = np.load( str(m) + '/'  + fix + '/CMx_' + fix + '.npy')
Y = np.load( str(m) + '/'  + fix + '/CMy_' + fix + '.npy')
Z = np.load( str(m) + '/'  + fix + '/CMz_' + fix + '.npy')
Mass = np.load( str(m) + '/'  + fix + '/Mass_' + fix + '.npy')
T = np.load( str(m) + '/'  + fix + '/T_' + fix + '.npy')
Den = np.load( str(m) + '/'  + fix + '/Den_' + fix + '.npy')
Rad = np.load( str(m) + '/'  +fix + '/Rad_' + fix + '.npy')

# Convert Energy / Mass to Energy Density in CGS
Rad *= Den 
Rad *= en_den_converter
Den *= den_converter 
# Convert to spherical
R, THETA, PHI = cartesian_to_spherical(X,Y,Z)
R = R.value 
THETA = THETA.value
PHI = PHI.value

#print('Den (weight) from simulation:', np.divide(np.sum(Den*Mass), np.sum(Mass)))

# Ensure that the regular grid cells are smaller than simulation cells
start = 2 * Rt
stop = 10_000 #400 * Rt
if m == 6:
    num = 750 + 1 # about the average of cell radius
if m == 4:
    num = 500 #350
radii = np.linspace(start, stop, num) #simulator units

# Find observers with Healpix
thetas = np.zeros(192)
phis = np.zeros(192)
observers = []
for i in range(0,192):
    thetas[i], phis[i] = hp.pix2ang(NSIDE, i)
    thetas[i] -= np.pi/2 # Enforce theta in -pi to pi
    
    observers.append( (thetas[i], phis[i]) )

# Cast
T_casted, Den_casted, Mass_casted = THROUPLE_S_CASTERS(radii, R, 
                                                   observers, THETA, PHI,
                                                   T, Den, Mass,
                                                   weights = Mass, 
                                                   avg = False)

# Clean
T_casted = np.nan_to_num(T_casted, neginf = 0)
Den_casted = np.nan_to_num(Den_casted, neginf = 0)
Mass_casted = np.nan_to_num(Mass_casted, neginf = 0)

# DROP THE LAST ONE
T_casted = np.delete(T_casted, -1, axis = 0)
Den_casted = np.delete(Den_casted, -1, axis = 0)
Mass_casted = np.delete(Mass_casted, -1, axis = 0)
radii = np.delete(radii, -1, axis = 0)
#%%
fig, ax = plt.subplots( figsize = (8,4) )
ax.axvline( np.log10(np.exp(-23)) , 
            color = 'b', linestyle = 'dashed')
ax.axvline( np.log10(np.exp(-0.18)) , 
            color = 'b', linestyle = 'dashed')
ax.axhline( np.log10(np.exp(8.666)) , 
            color = 'r', linestyle = 'dashed')
ax.axhline( np.log10(np.exp(17.876)) , 
            color = 'r', linestyle = 'dashed')
dens =  []
Ts = []
colors = []
masses = []
for i in range(len(radii)):
    for j in range(len(observers)):
        dens.append( np.log10( Den_casted[i,j]) )
        Ts.append( np.log10( T_casted[i,j] ))
        colors.append(radii[i])
        masses.append(np.log10(Mass_casted[i,j]))

kind = 'distance' # or dist
if kind == 'mass':
    img = plt.scatter(dens, Ts, s = 1, c = masses, marker = 'x', cmap = 'cet_bky')
    cbar = plt.colorbar(img)
    cbar.set_label(r'$\log_{10}(M)$ [$M_\odot$]')
elif kind == 'distance':
    img = plt.scatter(dens, Ts, s = 1, c = colors , marker = 'x', cmap = 'cet_bmy')
    cbar = plt.colorbar(img)
    cbar.set_label(r' Distance from BH $[R_\odot]$')

ax.grid()
ax.set_xlabel(r'$\log( \rho )$ $[g/cm^3]$')
ax.set_ylabel('$\log(T)$ $[K]$')