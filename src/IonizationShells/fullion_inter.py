#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:03:36 2024

@author: konstantinos
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.interpolate as sci

import src.Utilities.prelude as c
from src.Utilities.isalice import isalice
from src.Utilities.parser import parse



#%%
alice, plot = isalice()
if alice:
    pre = '/home/s3745597/data1/TDE/'
else:
    pre = ''
transition_slices = []

# Choose Simulation
if alice:
    args = parse()
    sim = args.name
    m = args.mass
    r = args.radius
    Mbh = args.blackhole
    fixes = np.arange(args.first, args.last + 1)
else:
    fixes = [164]
    sim = '4half'
    res = 'FID' # just for titling
    time = '0.5' # same 

argmax_fuckery = False
for fix in fixes:
    print(fix)
    if alice:
        X = np.load(f'{pre}{sim}/snap_{fix}/CMx_{fix}.npy')
        Y = np.load(f'{pre}{sim}/snap_{fix}/CMy_{fix}.npy')
        Z = np.load(f'{pre}{sim}/snap_{fix}/CMz_{fix}.npy')
        Den = np.load(f'{pre}{sim}/snap_{fix}/Den_{fix}.npy')
        Vol = np.load(f'{pre}{sim}/snap_{fix}/Vol_{fix}.npy')
        Vx = np.load(f'{pre}{sim}/snap_{fix}/Vx_{fix}.npy')
        Vy = np.load(f'{pre}{sim}/snap_{fix}/Vy_{fix}.npy')
        T = np.load(f'{pre}{sim}/snap_{fix}/T_{fix}.npy')
    else:
        X = np.load(f'{pre}{sim}/{fix}/CMx_{fix}.npy')
        Y = np.load(f'{pre}{sim}/{fix}/CMy_{fix}.npy')
        Z = np.load(f'{pre}{sim}/{fix}/CMz_{fix}.npy')
        Den = np.load(f'{pre}{sim}/{fix}/Den_{fix}.npy')
        Vol = np.load(f'{pre}{sim}/{fix}/Vol_{fix}.npy')
        Vx = np.load(f'{pre}{sim}/{fix}/Vx_{fix}.npy')
        Vy = np.load(f'{pre}{sim}/{fix}/Vy_{fix}.npy')
        T = np.load(f'{pre}{sim}/{fix}/T_{fix}.npy')
    # Mask
    mstar = 0.5
    rstar = 0.47
    mbh = 1e4
    Rt = rstar * (mbh / mstar)**(1/3) 

    xmin = -25*Rt
    xmax = 3.5*Rt
    ymin = -6*Rt
    ymax = 6*Rt

    rcell = np.power(Vol,1/3)
    midmask = np.where( (Z < 3 + rcell) &  (Z > -3 + rcell))[0]
    X = X[midmask]
    Y = Y[midmask]
    Z = Z[midmask]
    Den = Den[midmask]
    Vx = Vx[midmask]
    Vy = Vy[midmask]
    T = T[midmask]

    locmask = np.where((X > xmin) & (X < xmax) & (Y > ymin) & (Y < ymax) )[0]
    X = X[locmask]
    Y = Y[locmask]
    Z = Z[locmask]
    Den = Den[locmask]
    Vx = Vx[locmask]
    Vy = Vy[locmask]
    T = T[locmask]

    denmask = np.where((Den > 1e-12))[0]
    X = X[denmask]
    Y = Y[denmask]
    Z = Z[denmask]
    Vx = Vx[denmask]
    Vy = Vy[denmask]
    Den = Den[denmask]
    T = T[denmask]

    del denmask, locmask, midmask

    Vol *= c.Rsol_to_cm**3
    Den *= c.den_converter

    #%% Use table to get Ionization
    Ts = np.logspace(3, 11, num=100)
    Dens = np.logspace(-13, 2, num = 100)
    xH_tab = np.load('src/IonizationShells/xH.npy')
    xHe1_tab = np.load('src/IonizationShells/xHe1.npy')
    xHe2_tab = np.load('src/IonizationShells/xHe2.npy')

    hion = sci.RectBivariateSpline(Ts, Dens, xH_tab)
    heion1 = sci.RectBivariateSpline(Ts, Dens, xHe1_tab)
    heion2 = sci.RectBivariateSpline(Ts, Dens, xHe2_tab)
    
    xH = np.zeros(len(T))
    xHe1 = np.zeros(len(T))
    xHe2 = np.zeros(len(T))
    for i in range(len(xH)):
        xH[i] = hion(T[i], Den[i])
        xHe1[i] = heion1(T[i], Den[i])
        xHe2[i] = heion2(T[i], Den[i])
    #%% Ray Maker | This should be a class
    ray_no = 200
    thetas = np.linspace(-np.pi, np.pi, num = ray_no)
    THETA = np.arctan2(Y,X)
    R = np.sqrt(X**2 + Y**2)
    rays = [[] for _ in range(ray_no)]
    dens = [[] for _ in range(ray_no)]
    for i  in tqdm(range(len(R))):
        ray = np.argmin(np.abs(THETA[i]-thetas)) # could be faster with masks
        rays[ray].append({'idx':i, 'x':X[i], 'y':Y[i], 
                        'den':Den[i], 'z':Z[i], 'vx':Vx[i], 'vy':Vy[i],
                        'ion1':xH[i], 'ion2':xHe1[i], 'ion3':xHe2[i],
                        'T':T[i]})
        dens[ray].append(Den[i])
    # del X, Y, Z, Vx, Vy, P, T, rcell, Den
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
    #%% Find where the trensition happens.
    mean_xH = np.zeros(ray_no)
    density_maxima=np.array(density_maxima)
    for i in range(5, len(stream) -5):
        salami = np.array(stream[i]).T
        xH_in_salami = np.nan_to_num(salami[5], nan = 1) # I have manually checked and all the nans should be 1.
        mean_xH[i] = np.nan_to_num(np.mean(xH_in_salami), nan = -1) # argmax fuckery
    
        if mean_xH[i] > 0.01 and mean_xH[i] < 0.99:
            transition_slices.append( (i, mean_xH[i]) )
            
    if alice:
        np.savetxt(f'{pre}tde_comparison/data/ion{sim}.txt', transition_slices)

    if plot:
        plt.ioff()
        for check in range(ray_no):#ray_no) ):
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
            fig.suptitle(f'M$_*$ = 0.5 $M_\odot$ | M$_\mathrm{{BH}}$ = 10$^4 M_\odot$ | Res: {res} | t/t$_\mathrm{{fb}}$ = {time}')
            
            
            # Temperature
            cbar = axs[0,2].scatter( ns, zs,
                                c = np.log10(to_plot[8]), marker = 'h',
                                cmap='cet_CET_L4', s = msize, alpha = alpha,
                                vmin = 3.5, vmax = 4.5)
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
            
            figno = str(ray_no - check)
            if len(figno) < 2:
                figno = '0'+figno
            plt.savefig(f'/home/konstantinos/ionfig/{figno}.png', dpi = 300)
            plt.close()
