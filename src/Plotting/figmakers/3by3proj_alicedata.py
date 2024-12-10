#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:55:26 2024

@author: konstantinos
"""
# Vanilla
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import colorcet
import healpy as hp

# Choc
import src.Utilities.prelude as c

# Constants
rstar = 0.47
mstar = 0.5
Rt4 = rstar * (1e4/mstar)**(1/3)
Rt5 = rstar * (1e5/mstar)**(1/3)
Rt6 = rstar * (1e6/mstar)**(1/3)
 
def find_sph_coord(r, theta, phi):
    x = r * np.sin(np.pi-theta) * np.cos(phi) #Elad has just theta
    y = r * np.sin(np.pi-theta) * np.sin(phi)
    z = r * np.cos(np.pi-theta)
    return [x,y,z]

def equator_photo(rays_photo): 
    # rays_photo = rays_photo/c.Rsol_to_cm # to solar unit to plot
    
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
    # photo_x = np.append(photo_x, photo_x[0])
    # photo_y = np.append(photo_y, photo_y[0])
    
    return photo_x, photo_y

prepre = 'data/denproj/'
pre = 'R0.47M0.5BH'
suf = 'beta1S60n1.5Compton'
when = 'test' # choices: early mid late test
plane = 'XY'
if when == 'test':
    # 0.42, 0.82, 1.22
    fixes4 = [180, 297, 348] 
    fixes5 = [205, 285, 346] # 0.5, 1, 1.4
    fixes6 = [295, 376, 412] # 0.5, should be 376 and 414 but ok
    title_txt = 'Time: Trial t/t$_{FB}$'

fig, ax = plt.subplots(3,3, figsize = (12,4))
for f4, f5, f6, i in zip(fixes4, fixes5, fixes6, range(3)):
    # Load projection data
    den4 = np.loadtxt(f'{prepre}{pre}10000{suf}/denproj{pre}10000{suf}{f4}.txt')
    x4 = np.loadtxt(f'{prepre}{pre}10000{suf}/xarray{pre}10000{suf}.txt')
    y4 = np.loadtxt(f'{prepre}{pre}10000{suf}/yarray{pre}10000{suf}.txt')
    
    den5 = np.loadtxt(f'{prepre}{pre}100000{suf}/denproj{pre}100000{suf}{f5}.txt')
    x5 = np.loadtxt(f'{prepre}{pre}100000{suf}/xarray{pre}100000{suf}.txt')
    y5 = np.loadtxt(f'{prepre}{pre}100000{suf}/yarray{pre}100000{suf}.txt')
    
    den6 = np.loadtxt(f'{prepre}{pre}1e+06{suf}/denproj{pre}1e+06{suf}{f6}.txt')
    x6 = np.loadtxt(f'{prepre}{pre}1e+06{suf}/xarray{pre}1e+06{suf}.txt')
    y6 = np.loadtxt(f'{prepre}{pre}1e+06{suf}/yarray{pre}1e+06{suf}.txt')
    
    # Load photosphere
    photodata4 = np.genfromtxt('data/photosphere/tube_photo1.csv', delimiter = ',')
    photodata5 = np.genfromtxt('data/photosphere/photocolor5.csv', delimiter = ',')
    photodata6 = np.genfromtxt('data/photosphere/photocolor6.csv', delimiter = ',')
    
    # Find snap in photodata
    idx4 = np.argmin(np.abs(f4 - photodata4.T[0]))
    idx5 = np.argmin(np.abs(f5 - photodata5.T[0]))
    idx6 = np.argmin(np.abs(f6 - photodata6.T[0]))
    
    # snap time photo color obs_num
    photo_x4, photo_y4 = equator_photo(photodata4[idx4][4:4+192])
    photo_x5, photo_y5 = equator_photo(photodata5[idx5][4:4+192])
    photo_x6, photo_y6 = equator_photo(photodata6[idx6][4:4+192])

    # Plot projection data
    dmin = 0.1
    dmax = 5
    img = ax[i,0].pcolormesh(x4/Rt4, y4/Rt4, np.log10(den4.T), cmap = 'cet_fire',
                             vmin = dmin, vmax = dmax)
    ax[i,1].pcolormesh(x5/Rt5, y5/Rt5, np.log10(den5.T), cmap = 'cet_fire',
                             vmin = dmin, vmax = dmax)
    ax[i,2].pcolormesh(x6/Rt6, y6/Rt6, np.log10(den6.T), cmap = 'cet_fire',
                             vmin = dmin, vmax = dmax)
    
    # Plot Rt
    ax[i,0].add_patch(mp.Circle((0,0), Rt4/Rt4, ls = '-', 
                                color = 'c', fill = False, lw = 1))
    ax[i,1].add_patch(mp.Circle((0,0), Rt5/Rt5, ls = '-', 
                                color = 'c', fill = False, lw = 1))
    ax[i,2].add_patch(mp.Circle((0,0), Rt6/Rt6, ls = '-',
                                color = 'c', fill = False, lw = 1))
    
    if i == 0:
        ax[i,0].set_title('$10^4 M_\odot$', fontsize = 17)
        ax[i,1].set_title('$10^5 M_\odot$', fontsize = 17)
        ax[i,2].set_title('$10^6 M_\odot$', fontsize = 17)

        # ax[i,0].set_title('0.42 $t_\mathrm{FB}$', fontsize = 17)
        # ax[i,1].set_title('0.82 $t_\mathrm{FB}$', fontsize = 17)
        # ax[i,2].set_title('1.22 $t_\mathrm{FB}$', fontsize = 17)

    # Plot photosphere
    ax[i,0].plot(photo_x4 /Rt4, photo_y4/Rt4, '-o', 
                 c = 'magenta', markersize = 3)
    if i == 3:

        # ax[i,1].plot(photo_x5/Rt5, photo_y5/Rt5, '-o', 
        #              c = 'magenta', markersize = 1)
        # ax[i,2].plot(photo_x6/Rt6, photo_y6/Rt6, '-o', 
        #              c = 'magenta', markersize = 1)
        
        ax[i,0].add_patch(mp.Circle((0,0), photodata4[idx4][2]/Rt4, ls = '-', 
                                    color = 'm', fill = False, lw = 1))
        ax[i,1].add_patch(mp.Circle((0,0), photodata5[idx4][2]/Rt5, ls = '-', 
                                    color = 'm', fill = False, lw = 1))
        ax[i,2].add_patch(mp.Circle((0,0), photodata6[idx4][2]/Rt6, ls = '-',
                                    color = 'm', fill = False, lw = 1))

    # Set x-lims
    # 4
    
    
    
    # 5
    
    
    # 6
    ax[1,1].set_xlim(-50, 10)
    ax[1,1].set_ylim(-10, 20)

ax[2,1].set_xlabel('X $[R_\mathrm{T}]$', fontsize = 17)
ax[1,0].set_ylabel('Y $[R_\mathrm{T}]$', fontsize = 17)
cb = fig.colorbar(img, cax=fig.add_axes([0.93, 0.11, 0.03, 0.78]))
cb.set_label('$\log_{10} (\Sigma) $ [g/cm$^2$]', fontsize = 17)
# ax[3,2].set_xlabel('Y Coordinate')

    
    
    
