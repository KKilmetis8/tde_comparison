#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 16:22:40 2023

@author: konstantinos
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from astropy.coordinates import cartesian_to_spherical
from src.Opacity.opacity_table import opacity

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

fix = '881'
m = 6
X = np.load( str(m) + '/'  + fix + '/CMx_' + fix + '.npy')
Y = np.load( str(m) + '/'  + fix + '/CMy_' + fix + '.npy')
Z = np.load( str(m) + '/'  + fix + '/CMz_' + fix + '.npy')
Mass = np.load( str(m) + '/'  + fix + '/Mass_' + fix + '.npy')
T = np.load( str(m) + '/'  + fix + '/T_' + fix + '.npy')
Den = np.load( str(m) + '/'  + fix + '/Den_' + fix + '.npy')
Rad = np.load( str(m) + '/'  +fix + '/Rad_' + fix + '.npy')

Rad *= Den 
Rad *= en_den_converter
Den *= den_converter 
# Convert to spherical
R, THETA, PHI = cartesian_to_spherical(X,Y,Z)
# R = R.value 
THETA = THETA.value
PHI = PHI.value

#%% Find observers with Healpix
thetas = np.zeros(192)
phis = np.zeros(192)
observers = []
delta_thetas = []
delta_phis = []
fig, ax = plt.subplots(1,1, subplot_kw=dict(projection="mollweide"))
for i in range(192):
    theta_diff = 0
    phi_diff = 0
    
    thetas[i], phis[i] = hp.pix2ang(NSIDE, i)
    thetas[i] -= np.pi/2 # Enforce theta in -pi to pi    
    observers.append( (thetas[i], phis[i]) )
    
    corner_angles = []
    corner_x, corner_y, corner_z = hp.boundaries(4,i)
    corners = np.zeros((4,3))
    corner_angles = np.zeros((4,2))
    for i in range(4):
        corners[i] = corner_x[i], corner_y[i], corner_z[i]
        corner_angles[i][0] = hp.vec2ang(corners[i])[0][0]
        corner_angles[i][1] = hp.vec2ang(corners[i])[1][0]
        
    print(corner_angles.T[1].max() - corner_angles.T[1].min())
    corner_angles_phi_plot = list(corner_angles.T[1] - np.pi)
    corner_angles_phi_plot.append(corner_angles_phi_plot[0])
    corner_angles_theta_plot = list(corner_angles.T[0] - np.pi/2)
    corner_angles_theta_plot.append(corner_angles_theta_plot[0])
    
    plt.plot(corner_angles_phi_plot, corner_angles_theta_plot, 
             '-x', markersize = 1, c='k')
    # for i in range(0,4):
    #     theta_diff_temp = np.max(np.abs(corner_angles[i][0] - corner_angles[:][0]))
    #     phi_diff_temp = np.max(np.abs(corner_angles[i][1] - corner_angles[:][1])) 
        
    #     if theta_diff_temp > theta_diff:
    #         theta_diff = theta_diff_temp
            
    #     if phi_diff_temp > phi_diff:
    #         phi_diff = phi_diff_temp
            
    # delta_thetas.append(theta_diff/2)
    # delta_phis.append(phi_diff/2)
plt.plot(PHI[::100] - np.pi, THETA[::100], 'x',  c = 'r', markersize = 0.1)       
#%% Make rays

rays_T = []
rays_Den = []
rays_R = []
rays_Rad = []

for observer in observers:
    # These are kind of wrong, we should be more explicit about this
    delta_phi = 0.392 / 2 # rad
    delta_theta = 0.339 / 2 # rad
    # numpy thinks in C
    theta_mask = ( observer[0] - delta_theta < THETA) & \
                 ( THETA < observer[0] + delta_theta)
                 
    phi_mask = ( observer[1] - delta_phi < PHI) & \
               ( PHI < observer[1] + delta_phi)
    # NOTE: Mask, we need to sort by R
    rays_R.append( R[theta_mask & phi_mask])
    rays_Den.append( Den[theta_mask & phi_mask])
    rays_T.append( T[theta_mask & phi_mask])
    rays_Rad.append( Rad[theta_mask & phi_mask])
#%%
