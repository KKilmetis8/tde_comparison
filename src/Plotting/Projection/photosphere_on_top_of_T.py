#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 14:52:46 2024

@author: konstantinos
"""

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import src.Utilities.prelude as c
import src.Utilities.selectors as s
from src.Calculators.ray_forest import find_sph_coord, ray_finder, ray_maker_forest
from src.Luminosity.special_radii_tree import get_specialr
from src.Calculators.THREE_tree_caster import grid_maker

m = 6
num = 400
check = 'fid' # S60ComptonHires' #'S60ComptonHires'
snapshots, days = s.select_snap(m, check)
opacity_kind = s.select_opacity(m)
#%% Get Midplane
Mbh = 10**m 
Rt =  Mbh**(1/3) # Msol = 1, Rsol = 1
what = 'temperature'
gridded_indexes, grid_den, grid_mass, xs, ys, zs = grid_maker(snapshots[0], m, check, what,
                                                              False, 200, 200, 100)

#%% Get Photosphere    
for index in range(len(snapshots)): 
    snap = snapshots[index]       
    print('Snapshot ' + str(snap))
    filename = f"{m}/{snap}/snap_{snap}.h5"
    
    thetas, phis, stops = ray_finder(filename)
    rays = ray_maker_forest(snap, m, check, thetas, phis, stops, num, 
                            opacity_kind)
    

    _, _, rays_photo, _, _ = get_specialr(rays.T, rays.den, rays.radii, 
                                          rays.tree_indexes, opacity_kind, 
                                          select = 'photo' )
    
    _, _, rays_thermr, _, _ = get_specialr(rays.T, rays.den, rays.radii, 
                                          rays.tree_indexes, opacity_kind, 
                                          select = 'thermr_plot' )
    rays_photo = rays_photo/c.Rsol_to_cm # to solar unit to plot
    rays_thermr = rays_thermr/c.Rsol_to_cm
#%% Plot Midplane
import colorcet
fig, ax = plt.subplots(1,1)
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.figsize'] = [6, 4]
plt.rcParams['axes.facecolor']= 	'whitesmoke'
# Specify
if what == 'density':
    cb_text = r'Density [g/cm$^2$]'
    vmin = 0
    vmax = 7
elif what == 'temperature':
    cb_text = r'Temperature [K]'
    vmin = 3
    vmax = 7
else:
    raise ValueError('Hate to break it to you champ \n \
                     but we don\'t have that quantity')
        
den_plot = np.nan_to_num(grid_den, nan = -1, neginf = -1)
den_plot = np.log10(den_plot[:,:, len(zs)//2])
den_plot = np.nan_to_num(den_plot, nan = 0, neginf= 0)

ax.set_xlabel(r' X/$R_T$ [R$_\odot$]', fontsize = 14)
ax.set_ylabel(r' Y/$R_T$ [R$_\odot$]', fontsize = 14)
img = ax.pcolormesh(xs/Rt, ys/Rt, den_plot.T, cmap = 'cet_fire',
                    vmin = vmin, vmax = vmax)
cb = plt.colorbar(img)
cb.set_label(cb_text, fontsize = 14)
ax.set_title('Midplane', fontsize = 16)
ax.set_xlim(-40,)
ax.set_ylim(-30,)

# Plot the ones in the equatorial plane
photo_x = []
photo_y = []
therm_x = []
therm_y = []
for iobs in range(0,192):
    theta, phi = hp.pix2ang(4, iobs) # theta in [0,pi], phi in [0,2pi]
    r_ph = rays_photo[iobs]
    xyz_ph = find_sph_coord(r_ph, theta, phi)
    if np.abs(xyz_ph[2]) < 2: # Equatorial plane |z| < 2
        photo_x.append(xyz_ph[0])
        photo_y.append(xyz_ph[1])
    r_th = rays_thermr[iobs]    
    xyz_th = find_sph_coord(r_th, theta, phi)
    if np.abs(xyz_th[2]) < 2:
        # print(xyz_th)
        therm_x.append(xyz_th[0])
        therm_y.append(xyz_th[1])

photo_x = np.array(photo_x)
photo_y = np.array(photo_y)
therm_x = np.array(therm_x)
therm_y = np.array(therm_y)
ax.plot(photo_x / Rt, photo_y / Rt, 
        marker = 'o', color = 'springgreen', 
        linewidth = 1)
ax.plot(therm_x / Rt, therm_y / Rt, 
         marker = 's', color = 'b', 
         linestyle = '-.', linewidth = 1)