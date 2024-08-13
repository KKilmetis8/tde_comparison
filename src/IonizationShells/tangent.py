#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 12:06:45 2024

@author: konstantinos
"""

# Vanilla
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from tqdm import tqdm

# Chocolate
import src.Utilities.prelude as c

# NOTE: TRACK THE NUMBER OF CELLS IN THE STREAM
#%% Data Import
fix = 164
sim = '4halfHR'
X = np.load(f'{sim}/{fix}/CMx_{fix}.npy')
Y = np.load(f'{sim}/{fix}/CMy_{fix}.npy')
Z = np.load(f'{sim}/{fix}/CMz_{fix}.npy')
Den = np.load(f'{sim}/{fix}/Den_{fix}.npy')
Vol = np.load(f'{sim}/{fix}/Vol_{fix}.npy')
Vx = np.load(f'{sim}/{fix}/Vx_{fix}.npy')
Vy = np.load(f'{sim}/{fix}/Vy_{fix}.npy')
T = np.load(f'{sim}/{fix}/T_{fix}.npy')
P = np.load(f'{sim}/{fix}/P_{fix}.npy')

# Mask
mstar = 0.5
rstar = 0.47
mbh = 1e4
Rt = rstar * (mbh / mstar)**(1/3) 

xmin = -15*Rt
xmax = 2.5*Rt
ymin = -3*Rt
ymax = 3*Rt

rcell = np.power(Vol,1/3)
midmask = np.where( (Z < 3 + rcell) &  (Z > -3 + rcell))[0]
X = X[midmask]
Y = Y[midmask]
Z = Z[midmask]
rcell = rcell[midmask]
Den = Den[midmask]
Vx = Vx[midmask]
Vy = Vy[midmask]
P = P[midmask]
T = T[midmask]

locmask = np.where((X > xmin) & (X < xmax) & (Y > ymin) & (Y < ymax) )[0]
X = X[locmask]
Y = Y[locmask]
Z = Z[locmask]
rcell = rcell[locmask]
Den = Den[locmask]
Vx = Vx[locmask]
Vy = Vy[locmask]
P = P[locmask]
T = T[locmask]

denmask = np.where((Den > 1e-12))[0]
X = X[denmask]
Y = Y[denmask]
Z = Z[denmask]
rcell = rcell[denmask]
Vx = Vx[denmask]
Vy = Vy[denmask]
Den = Den[denmask]
P = P[denmask]
T = T[denmask]

vol = rcell**3
Mass = Den*vol
Mass = np.log10(Mass)
# Den = np.log10(Den)

del denmask, locmask, midmask
#%% Calc. Ion fraction.
pressure_converter = c.Msol_to_g / (c.Rsol_to_cm * c.t**2)
P *=  pressure_converter
Den *= c.den_converter

# Caclculate Ks
# NOTE: Add degeneracy factors
K1 = c.prefactor_h * (2*np.pi/c.me)**1.5 * c.hbar**3 * np.exp(c.xh/(c.kb * T)) / (c.kb * T**2.5)
ion1 = np.divide(1, np.sqrt(1 + P*K1))

K2 = c.prefactor_he1 * (2*np.pi/c.me)**1.5 * c.hbar**3 * np.exp(c.xhe1/(c.kb * T)) / (c.kb * T**2.5)
ion2 = np.divide(1, np.sqrt(1 + P*K2))

K3 = c.prefactor_he2 * (2*np.pi/c.me)**1.5 * c.hbar**3 * np.exp(c.xhe2/(c.kb * T)) / (c.kb * T**2.5)
ion3 = np.divide(1, np.sqrt(1 + P*K3))

del K1, K2, K3
#%% Sanity Plot
# plt.figure(figsize=(5,3.75))
# step = 10
# plt.scatter(X[::step]/Rt,Y[::step]/Rt, c=Den[::step], 
#             s = rcell[::step], )
# plt.xlim(xmin/Rt, xmax/Rt)
# plt.ylim(ymin/Rt, ymax/Rt)
# plt.xlabel('X [$R_\odot$]', fontsize = 14)
# plt.ylabel('Y [$R_\odot$]', fontsize = 14)
#%% Ray Maker | This should be a class
ray_no = 100
thetas = np.linspace(-np.pi, np.pi, num = ray_no)
THETA = np.arctan2(Y,X)
R = np.sqrt(X**2 + Y**2)
rays = [[] for _ in range(ray_no)]
dens = [[] for _ in range(ray_no)]
for i  in tqdm(range(len(R))):
    ray = np.argmin(np.abs(THETA[i]-thetas)) # could be faster with masks
    rays[ray].append({'idx':i, 'x':X[i], 'y':Y[i], 
                      'den':Den[i], 'z':Z[i], 'vx':Vx[i], 'vy':Vy[i],
                      'ion1':ion1[i], 'ion2':ion2[i], 'ion3':ion3[i],
                      'T':T[i]})
    dens[ray].append(Den[i])
del X, Y, Z, Mass, Vx, Vy, P, T, rcell, Den
#%% Density maximum
density_maxima = [[] for _ in range(ray_no)]
stream = [[] for _ in range(ray_no)]

for i in tqdm( range(ray_no)):
    # ray_array = np.array(rays[i])
    den_maxidx = np.argmax(dens[i])
    
    den_max_point = rays[i][den_maxidx]
    # If I have velocity (vx, vy), the normal vector is n = (vy, -vx) 
    # to point in and n = (-vy, vx) to point out 
    mag = np.sqrt(den_max_point['vx']**2+den_max_point['vy']**2)
    that = (den_max_point['vx'] / mag, den_max_point['vy'] / mag) 
    nhat = (-den_max_point['vy'] /mag, den_max_point['vx'] / mag) 
    
    t_coord = np.dot( [ den_max_point['x'], den_max_point['y']], that)
    n_coord = np.dot( [ den_max_point['x'], den_max_point['y']], nhat)
    
    density_maxima[i].append((den_max_point['x'], den_max_point['y'], 
                              den_max_point['z'], n_coord))

    # Plot
    # plt.xlabel('X [$R_\odot$]', fontsize = 14)
    # plt.ylabel('Y [$R_\odot$]', fontsize = 14)
    # plt.scatter(den_max_point['x'], den_max_point['y'])
    # plt.arrow(den_max_point['x'], den_max_point['y'], den_max_point['vx'], den_max_point['vy'], 
    #           width = 0.2, color= 'k')
    # plt.arrow(den_max_point['x'], den_max_point['y'], den_max_point['vy'], -den_max_point['vx'], 
    #           width = 0.2, color= 'r')
    # plt.text(den_max_point['x'], den_max_point['y'], str(i))
    for j, cell in enumerate(rays[i]): 
        if cell['den']  > 1/3 * den_max_point['den']: # criterion
            # dist = np.linalg.norm( np.dot( [ cell['x'], cell['y']], that))
            # dist = np.abs(dist - t_coord)
            # print(dist)
            # if dist<5: # be close to den max, tangent-wise
            x_from_denmax = cell['x'] - den_max_point['x']
            y_from_denmax = cell['y'] - den_max_point['y']
            n_coord = np.dot([x_from_denmax, y_from_denmax], nhat)
            stream[i].append((j, cell['z'], n_coord, x_from_denmax, y_from_denmax, 
                              cell['ion1'], cell['ion2'], cell['ion3'],  
                              cell['T'], cell['den']))

#%% Plot stream sections

plt.ioff()
for check in tqdm( range(15, 90)):#ray_no) ):
    if len(density_maxima[check]) < 1 or len(stream[check]) < 1:
        continue
    
    to_plot = np.array(stream[check]).T
    center = density_maxima[check][0]
    center_n = center[3]
    center_z = center[2]

    ns = to_plot[2]# - center_n
    zs = to_plot[1]# - center_z
    center_n = 0
    center_z = 0
    
    fig, axs = plt.subplots(2,3, figsize= (10,6), tight_layout = True)
    msize = 100 # len(to_plot[2])//30
    alpha = 0.8
    ymax = 3.5
    ymin = -3.5
    xmax = 3.5
    xmin = -3.5
    # Density
    cbar = axs[0,0].scatter(ns, zs,
                        c = to_plot[9], marker = 'h',
                        cmap='viridis',  s = msize,
                        vmin = 1e-7, vmax = 3e-6, alpha = alpha)
    axs[0,0].scatter(center_n, center_z, 
                c = c.AEK, marker = 'X', ec = 'k',
                s = 100, alpha = 0.3)
    cb = plt.colorbar(cbar)
    cb.set_label('Density [g/cm$^3$]')
    axs[0,0].set_xlabel('Width [$R_\odot$]', fontsize = 14)
    axs[0,0].set_ylabel('Height (Z) [$R_\odot$]', fontsize = 14)
    axs[0,0].set_ylim(ymin, ymax)
    axs[0,0].set_xlim(xmin, xmax)
    
    # Height
    h1 = np.argmax(zs)
    h2 = np.argmin(zs)
    w1 = np.argmax(ns)
    w2 = np.argmin(ns)
    
    axs[0,0].plot( (0,0),  (zs[h1],zs[h2]), c = 'r')
    axs[0,0].plot( (ns[w1],ns[w2]), (0,0), c = 'magenta')
    axs[0,0].text( 0.55,-3.2, f'H: {np.abs(zs[h1]-zs[h2]):.2f}',c='r', 
                  fontsize = 18 )
    axs[0,0].text( -3,-3.2, f'W: {np.abs(ns[w1]-ns[w2]):.2f}', c='magenta',
                  fontsize = 18 )

    num = len(zs)
    
    axs[0,0].scatter(center_n, center_z, 
                c = c.AEK, marker = 'X', ec = 'k',
                s = 100, alpha = 0.4)

    # Stream
    density_maxima=np.array(density_maxima)
    axs[0,1].scatter(density_maxima.T[0], density_maxima.T[1], 
                   c='k', marker = 'h')
    axs[0,1].scatter(center[0], center[1], 
                   c=c.AEK, marker = 'X', ec='k', s = 100, zorder = 3, alpha = 0.9)
    axs[0,1].set_xlabel('X [$R_\odot$]', fontsize = 14)
    axs[0,1].set_ylabel('Y [$R_\odot$]', fontsize = 14)
    
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # fig.text(0.65, 0.5, 'M$_*$ = 0.5 $M_\odot$ \n M$_\mathrm{BH}$ = 10$^4 M_\odot$ \n Res: fiducial  \n t/t$_\mathrm{fb}$ = 0.7',
    #          bbox = props)
    fig.suptitle('M$_*$ = 0.5 $M_\odot$ | M$_\mathrm{BH}$ = 10$^4 M_\odot$ | Res: HR | t/t$_\mathrm{fb}$ = 0.7')
    
    
    # Temperature
    cbar = axs[0,2].scatter( ns, zs,
                        c = np.log10(to_plot[8]), marker = 'h',
                        cmap='cet_fire', s = msize, alpha = alpha,
                        vmin = 4, vmax = 6.3)
    axs[0,2].scatter(center_n - center_n, center[2], 
                c = c.AEK, marker = 'X', ec = 'k',
                s = 100, alpha = 0.5)
    cb = plt.colorbar(cbar)
    cb.set_label('Log(Temperature) [K]')
    axs[0,2].set_xlabel('Width [$R_\odot$]', fontsize = 14)
    axs[0,2].set_ylabel('Height (Z) [$R_\odot$]', fontsize = 14)
    axs[0,2].set_ylim(ymin, ymax)
    axs[0,2].set_xlim(xmin, xmax)
    #axs[0,2].plot(np.log10(to_plot[8]))
    
    # Hyd Ion
    cbar = axs[1,0].scatter( ns, zs,
                        c = to_plot[5], marker = 'h', alpha = alpha,
                        cmap='cet_linear_grey_0_100_c0', ec = 'k', s = msize,
                        vmin = 0, vmax = 1)
    axs[1,0].scatter(center_n - center_n, center[2], 
                c = c.AEK, marker = 'X', ec = 'k',
                s = 100, alpha = 0.3)
    cb = plt.colorbar(cbar)
    cb.set_label('Hydrogen Ionization Fraction')
    axs[1,0].set_xlabel('Width [$R_\odot$]', fontsize = 14)
    axs[1,0].set_ylabel('Height (Z) [$R_\odot$]', fontsize = 14)
    axs[1,0].set_ylim(ymin, ymax)
    axs[1,0].set_xlim(xmin, xmax)
    
    # He Ion 1
    cbar = axs[1,1].scatter( ns, zs,
                        c = to_plot[6], marker = 'h', alpha = alpha,
                        cmap='cet_linear_grey_0_100_c0', ec = 'k', s = msize,
                        vmin = 0, vmax = 1)
    axs[1,1].scatter(center_n - center_n, center[2], 
                c = c.AEK, marker = 'X', ec = 'k',
                s = 100, alpha = 0.3)
    cb = plt.colorbar(cbar)
    cb.set_label('First Helium Ionization Fraction')
    axs[1,1].set_xlabel('Width [$R_\odot$]', fontsize = 14)
    axs[1,1].set_ylabel('Height (Z) [$R_\odot$]', fontsize = 14)
    axs[1,1].set_ylim(ymin, ymax)
    axs[1,1].set_xlim(xmin, xmax)
    
    # He Ion 2
    cbar = axs[1,2].scatter( ns, zs,
                        c = to_plot[7], marker = 'h', alpha = alpha,
                        cmap='cet_linear_grey_0_100_c0', ec = 'k', s = msize,
                        vmin = 0, vmax = 1)
    axs[1,2].scatter(center_n - center_n, center[2], 
                c = c.AEK, marker = 'X', ec = 'k',
                s = 100, alpha = 0.3)
    cb = plt.colorbar(cbar)
    cb.set_label('Second Helium Ionization Fraction')
    axs[1,2].set_xlabel('Width [$R_\odot$]', fontsize = 14)
    axs[1,2].set_ylabel('Height (Z) [$R_\odot$]', fontsize = 14)
    axs[1,2].set_ylim(ymin, ymax)
    axs[1,2].set_xlim(xmin, xmax)
    
    figno = str(97 - check)
    if len(figno) < 2:
        figno = '0'+figno
    plt.savefig(f'/home/konstantinos/ionfig/{figno}.png', dpi = 300)
    plt.close()

