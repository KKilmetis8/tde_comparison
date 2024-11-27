#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 15:14:37 2024

@author: konstantinos
"""
# Vanilla
import os
import numpy as np
import matplotlib.pyplot as plt
import colorcet
from scipy.spatial import KDTree
from tqdm import tqdm
import healpy as hp

# Choc
import src.Utilities.prelude as c
from src.Calculators.casters import THE_CASTER
from src.Calculators.ray_forest import find_sph_coord, ray_finder, ray_maker_forest
from src.Luminosity.special_radii_tree import get_specialr
from src.Calculators.THREE_tree_caster import grid_maker

#%%
def maker(m, pixel_num, fix, plane, thing, how, star = 'half'):
    Mbh = 10**m
    if 'star' == 'half':
        mstar = 0.5
        rstar = 0.47
    else:
        mstar = 1
        rstar = 1
    Rt = rstar * (Mbh/mstar)**(1/3) 
    apocenter = 2 * Rt * (Mbh/mstar)**(1/3)
    pre = f'{m}/{fix}'
    
    # Mass = np.load(pre + '/Mass_' + fix + '.npy')
    Den = np.load(pre + '/Den_' + fix + '.npy')
    # Need to convert Msol/Rsol^2 to g/cm
    Msol_to_g = 1.989e33
    Rsol_to_cm = 6.957e10
    converter = Msol_to_g / Rsol_to_cm**2
    Den *=  converter

    # CM Position Data
    X = np.load(pre + '/CMx_' + fix + '.npy')
    Y = np.load(pre + '/CMy_' + fix + '.npy')
    x_start = 2 * -apocenter
    x_stop = 0.2 * apocenter
    x_num = pixel_num # np.abs(x_start - x_stop)
    xs = np.linspace(x_start, x_stop, num = x_num )
    y_start = -0.4 * apocenter 
    y_stop = 0.4 * apocenter
    y_num = pixel_num # np.abs(y_start - y_stop)
    ys = np.linspace(y_start, y_stop, num = y_num)
    
    if how == 'caster':
        den_cast = THE_CASTER(xs, X, ys, Y, Den) # EVOKE

    # Make a tree
    if how == 'tree':
        Z = np.load(pre + '/CMz_' + fix + '.npy')

        sim_value = [X, Y, Z] 
        sim_value = np.transpose(sim_value) #array of dim (number_points, 3)
        sim_tree = KDTree(sim_value) 
        
        z_start = -2 * Rt
        z_stop = 2 * Rt
        z_num = 100
        z_radii = np.linspace(z_start, z_stop, z_num) #simulator units

        gridded_indexes =  np.zeros(( len(xs), len(ys), len(z_radii) ))
        den_cast =  np.zeros(( len(xs), len(ys), len(z_radii) ))
        gridded_mass =  np.zeros(( len(xs), len(ys), len(z_radii) ))
        for i in tqdm(range(len(xs))):
            for j in range(len(ys)):
                for k in range(len(z_radii)):
                    queried_value = [xs[i], ys[j], z_radii[k]]
                    _, idx = sim_tree.query(queried_value)
                                        
                    # Store
                    gridded_indexes[i, j, k] = idx
                    den_cast[i, j, k] = Den[idx]
        #             gridded_mass[i,j, k] = Mass[idx]
        # den_cast = np.divide(den_cast, gridded_mass)
        den_cast = np.sum(den_cast, axis=2) / 100

    # Remove bullshit and fix things
    den_cast = np.nan_to_num(den_cast.T)
    den_cast = np.log10(den_cast) # we want a log plot
    den_cast = np.nan_to_num(den_cast, neginf=0) # fix the fuckery
        
    # Color re-normalization
    #if thing == 'Den':
    if how == 'caster':
        den_cast[den_cast<1] = 0
        den_cast[den_cast>8] = 8
    if how == 'tree':
        den_cast[den_cast<0.2] = 0
        den_cast[den_cast>5] = 5

    return xs/apocenter, ys/apocenter, den_cast, apocenter# , days


def photo_plot(m, num, snap, opacity_kind, beta, star = 'half', photo = None):
    if type(photo) == type(None):
        filename = f"{m}/{snap}/snap_{snap}.h5"
        thetas, phis, stops, _ = ray_finder(filename)
        check = ''
        rays = ray_maker_forest(snap, m, star, check, thetas, phis, stops, num, 
                                opacity_kind, beta)
        
    
        _, _, rays_photo, _, _ = get_specialr(rays.T, rays.den, rays.radii, 
                                              rays.tree_indexes, opacity_kind, 
                                              select = 'photo' )        
        rays_photo = rays_photo/c.Rsol_to_cm # to solar unit to plot
    else:
        rays_photo = photo
        print('new')
    print(rays_photo)
    # Make zs
    Zs = []
    for iobs in range(0,192):
        theta, phi = hp.pix2ang(4, iobs) # theta in [0,pi], phi in [0,2pi]
        r_ph = rays_photo[iobs]
        xyz_ph = find_sph_coord(r_ph, theta, phi)
        Zs.append(xyz_ph[2])
    
    # Sort and keep 16 closest to equator
    idx_z_sorted = np.argsort(np.abs(Zs))
    
    size = 16
    plus_x = []
    neg_x = []
    plus_y = []
    neg_y = []
    photo_x = []
    photo_y = []
    
    for j, iz in enumerate(idx_z_sorted):
        if j == size:
            break
        theta, phi = hp.pix2ang(4, iz) # theta in [0,pi], phi in [0,2pi]
        r_ph = rays_photo[iz]
        xyz_ph = find_sph_coord(r_ph, theta, phi)
        if xyz_ph[1]>0:
            plus_x.append(xyz_ph[0])
            plus_y.append(xyz_ph[1])
        else:
            neg_x.append(xyz_ph[0])
            neg_y.append(xyz_ph[1])
    
    # Arrays so they can be indexed
    plus_x = np.array(plus_x)
    plus_y = np.array(plus_y)
    neg_x = np.array(neg_x)
    neg_y = np.array(neg_y)
    
    # Sort to untangle
    theta_plus = np.arctan2(plus_y,plus_x)
    cos_plus = np.cos(theta_plus)
    psort = np.argsort(cos_plus)
    
    theta_neg = np.arctan2(neg_y,neg_x)
    cos_neg = np.cos(theta_neg)
    nsort = np.argsort(cos_neg)

    # Combine correct
    photo_x = np.concatenate((plus_x[psort],  np.flip(neg_x[nsort],  )))
    photo_y = np.concatenate((plus_y[psort],  np.flip(neg_y[nsort], )))
    
    # Close the loop
    photo_x = np.append(photo_x, photo_x[0])
    photo_y = np.append(photo_y, photo_y[0])
    
    return photo_x, photo_y
#%%
plane = 'XY'
thing = 'Den' # Den or T
when = 'trial' # early mid late last
if when == 'trial':
    fixes4 = ['297'] # 0.65
    fixes5 = ['365'] # 0.55
    fixes6 = ['683'] # 0.5
    title_txt = 'Time: Trial t/t$_{FB}$'

# for i in range(len(fixes4)):

#     # Plotting
#     fig, ax = plt.subplots(3, 3, clear = True, tight_layout = True)
    
#     # Image making
#     x4, y4, d4 = maker(4, 500, fixes4[i], plane, thing)
#     ax[0,0].pcolormesh(x4, y4, d4, cmap='cet_fire', vmin = 0, vmax = 8)
#     del x4, y4, d4
#     x5, y5, d5 = maker(5, 500, fixes5[i], plane, thing)
#     img2 = ax[0,1].pcolormesh(x5, y5, d5, cmap='cet_fire', vmin = 0, vmax = 8)
#     del x5, y5, d5
#     x6, y6, d6 = maker(6, 500, fixes6[i], plane, thing)
#     ax[0,2].pcolormesh(x6, y6, d6, cmap='cet_fire', vmin = 0, vmax = 8)
#     del x6, y6, d6

i = 0
how = 'tree'
if how == 'tree':
    res = 200
else:
    res = 1000
    
photo_x4, photo_y4 = photo_plot(4, res, fixes4[i], 'LTE', 1,   photo = photosphere)
# photo_x5, photo_y5 = photo_plot(5, res, fixes5[i], 'LTE', 1)
# photo_x6, photo_y6 = photo_plot(6, res, fixes6[i], 'cloudy', 1)
#%%
x4, y4, d4, apo4 = maker(4, res, fixes4[i], plane, thing, how)
# x5, y5, d5, apo5 = maker(5, res, fixes5[i], plane, thing, how)
# x6, y6, d6, apo6 = maker(6, res, fixes6[i], plane, thing, how)

#%% Plotting
# rate = 2
# y = 5
# x = y * rate
# fig, ax = plt.subplots(3, 3, clear = True, tight_layout = True, 
#                        figsize = (x,y), sharex=True, sharey=True)
# if how == 'tree':
#     vmax = 5
# else:
#     vmax = 8
# # Image making ----------------------------------------------------------------
# width = 0.035
# ax[0,0].pcolormesh(x4, y4, d4, cmap='cet_fire', vmin = 0, vmax = vmax)
# img2 = ax[0,1].pcolormesh(x5, y5, d5, cmap='cet_fire', vmin = 0, vmax = vmax)
# ax[0,2].pcolormesh(x6, y6, d6, cmap='cet_fire', vmin = 0, vmax = vmax)

# # Photosphere -----------------------------------------------------------------
# ax[0,0].plot(photo_x4 / apo4, photo_y4 / apo4, 
#         marker = '', color = 'cyan', ls = '-',
#         linewidth = 2, markersize = 10)
# ax[0,1].plot(photo_x5 / apo5, photo_y5 / apo5, 
#         marker = '', color = 'cyan', ls = '-',
#         linewidth = 2, markersize = 10)
# ax[0,2].plot(photo_x6 / apo6, photo_y6 / apo6, 
#         marker = '', color = 'cyan', ls = '-',
#         linewidth = 1)
# # Limits
# ax[0,0].set_xlim(-1.,0.2)
# ax[0,0].set_ylim(-0.2,0.2)

# # Colorbar --------------------------------------------------------------------
# width = 0.035
# pad = 0.1
# cax = fig.add_axes([ax[1,2].get_position().xmax + pad, 
#                     ax[2,2].get_position().ymin, 
#                     width,
#                     ax[0,2].get_position().ymax - ax[2,2].get_position().ymin + 0.08]) 

# cb = fig.colorbar(img2, cax=cax)  
# cb.ax.tick_params(labelsize=25, pad = 5)
# cb.set_label(r'Density $\log_{10}(\rho)$ [g/cm$^2$]', fontsize = 20, labelpad = 15)
# # Axis labels
# # fig.text(0.5, -0.03, plane[0] + r' [x/R$_a$]', ha='center', fontsize = 25)
# # fig.text(-0.02, 0.5, plane[1] + r' [y/R$_a$]', va='center', rotation='vertical', fontsize = 25)
# ax[2,1].set_xlabel(plane[0] + r' [x/R$_a$]', ha='center', fontsize = 25 )
# ax[1,0].set_ylabel(plane[1] + r' [y/R$_a$]', ha='center', fontsize = 25 )

# #plt.savefig('xyproj.png')#, format = 'pdf', dpi = 400)

# from src.Utilities.finished import finished
# finished()
#%% Just 3
rate = 1.5
y = 5
x = y * rate
fig, ax = plt.subplots(1, 1, clear = True, tight_layout = True, 
                       figsize = (x,y), sharex=True, sharey=True)
if how == 'tree':
    vmax = 5
else:
    vmax = 8
# Image making ----------------------------------------------------------------
width = 0.035
ax.pcolormesh(x4, y4, d4, cmap='cet_fire', vmin = 0, vmax = vmax)
# img2 = ax[1].pcolormesh(x5, y5, d5, cmap='cet_fire', vmin = 0, vmax = vmax)
# ax[2].pcolormesh(x6, y6, d6, cmap='cet_fire', vmin = 0, vmax = vmax)

# Photosphere -----------------------------------------------------------------
img1 = ax.plot(photo_x4 / apo4, photo_y4 / apo4, 
        marker = 'o', color = 'cyan', ls = '-',
        linewidth = 2, markersize = 10)
# ax[1].plot(photo_x5 / apo5, photo_y5 / apo5, 
#         marker = 'o', color = 'cyan', ls = '-',
#         linewidth = 2, markersize = 10)
# ax[2].plot(photo_x6 / apo6, photo_y6 / apo6, 
#         marker = '', color = 'cyan', ls = '-',
#         linewidth = 1)
# Limits
# ax[0].set_xlim(-1.,0.2)
# ax[0].set_ylim(-0.2,0.2)

# Colorbar --------------------------------------------------------------------
left, bottom, width, height = 1, 0.14, 0.035, 0.82
cax = fig.add_axes([left, bottom, width, height]) 
cb = fig.colorbar(img1, cax=cax)  
cb.ax.tick_params(labelsize=25, pad = 5)
cb.set_label(r'Density $\log_{10}(\rho)$ [g/cm$^2$]', fontsize = 20, labelpad = 15)
# Axis labels
ax.text(-1.55, 0.2, r'10$^4 M_\odot$', ha='right', fontsize = 25, color = 'white')
# ax[1].text(-1.55, 0.2, r'10$^5 M_\odot$', ha='right', fontsize = 25, color = 'white')
# ax[1].set_xlabel(plane[0] + r' [x/R$_a$]', ha='center', fontsize = 25 )
# ax[1].set_ylabel(plane[1] + r' [y/R$_a$]', ha='center', fontsize = 25 )
ax.set_ylabel(plane[1] + r' [y/R$_a$]', ha='center', fontsize = 25 )

#plt.savefig('xyproj.png')#, format = 'pdf', dpi = 400)
    

