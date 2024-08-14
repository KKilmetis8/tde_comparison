#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 12:10:03 2024

@author: konstantinos
"""

# Vanilla
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.optimize import bisect
from tqdm import tqdm

# Chocolate
import src.Utilities.prelude as c
from src.Utilities.isalice import isalice
from src.Utilities.parser import parse
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
        P = np.load(f'{pre}{sim}/snap_{fix}/P_{fix}.npy')
    else:
        X = np.load(f'{pre}{sim}/{fix}/CMx_{fix}.npy')
        Y = np.load(f'{pre}{sim}/{fix}/CMy_{fix}.npy')
        Z = np.load(f'{pre}{sim}/{fix}/CMz_{fix}.npy')
        Den = np.load(f'{pre}{sim}/{fix}/Den_{fix}.npy')
        Vol = np.load(f'{pre}{sim}/{fix}/Vol_{fix}.npy')
        Vx = np.load(f'{pre}{sim}/{fix}/Vx_{fix}.npy')
        Vy = np.load(f'{pre}{sim}/{fix}/Vy_{fix}.npy')
        T = np.load(f'{pre}{sim}/{fix}/T_{fix}.npy')
        P = np.load(f'{pre}{sim}/{fix}/P_{fix}.npy')
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
    rcell = rcell[midmask]
    Den = Den[midmask]
    Vx = Vx[midmask]
    Vy = Vy[midmask]
    P = P[midmask]
    T = T[midmask]
    Vol = Vol[midmask]

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
    Vol = Vol[locmask]

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
    Vol = Vol[denmask]

    #vol = rcell**3
    #Mass = Den*vol
    # Mass = np.log10(Mass)
    # Den = np.log10(Den)

    del denmask, locmask, midmask

    Vol *= c.Rsol_to_cm**3
    Den *= c.den_converter

    #%% Get electron density by following Tomida+18

    def bisection(f, a, b, *args,
                tolerance=1e-2,
                max_iter=100):
        rf_step = 0
        # Sanity
        fa = f(a, *args)
        fb = f(b, *args)
        if fa * fb > 0:
            print('No roots here bucko')
            return a
        while rf_step < max_iter:
            rf_step += 1
            c = 0.5 * (a + b)
            fc = f(c, *args)
            if np.abs(fc) < tolerance:
                break
            if np.sign(fc) == np.sign(fa):
                a = c
            else:
                b = c
        # print('B steps: ', rf_step)
        return c

    class partition:
        # Straight out of Tomida
        def __init__(self, T, V):
            # Molecular Hydrogen
            Ztr_H2 = (2 * np.pi * c.mh2 * c.kb * T)**(3/2) / c.h**3
            Zrot_even_H2 = 0 
            Zrot_odd_H2 = 0
            for i in range(0,15,2):
                odd = i+1
                even = i
                Zrot_even_H2 += (2*even+1) * np.exp(- even * (even + 1) * c.rot / (2*T)) 
                Zrot_odd_H2 += (2*odd+1) * np.exp(- odd * (odd + 1) * c.rot / (2*T))
            Zrot = Zrot_even_H2**(1/4) * (3*Zrot_odd_H2 * np.exp(c.rot/T))**(3/4)
    #        Zvib_H2 = 1 / ( 2 * np.sinh(c.vib / (2*T)) ) # CHANGE THIS
            Zvib_H2 = 1 / (1 -  np.exp(c.vib/T))

            Zspin_H2 = 4 
            Zelec_H2 = 2
            self.Z_H2 = V * Ztr_H2 * Zrot * Zvib_H2 * Zspin_H2 * Zelec_H2
            
            # Atomic Hydrogen
            Ztr_H = (2 * np.pi * c.mh * c.kb * T)**(3/2) / c.h**3
            Zspin_H = 2
            Zelec_H = 2 * np.exp( - c.xdis_h2 / (2 * c.kb * T))
            self.Z_H = V * Ztr_H * Zspin_H * Zelec_H

            # Ionized Hydrogen
            Ztr_Hion = (2 * np.pi * c.mh * c.kb * T)**(3/2) / c.h**3
            Zspin_Hion = 2
            Zelec_Hion = 2 * np.exp( - (c.xdis_h2 + 2*c.xh) / (2 * c.kb * T))
            self.Z_Hion = V * Ztr_Hion * Zspin_Hion * Zelec_Hion
            
            # Atomic Helium
            Ztr_He = (2 * np.pi * c.mhe * c.kb * T)**(3/2) / c.h**3
            self.Z_He = V * Ztr_He
            
            # 1 Ionized Helium 
            # Ztr_He1 = 2 * np.pi * c.mhe * c.kb * T**(3/2) / c.h**3
            Zelec_He1 = np.exp( - c.xhe1 / ( c.kb * T))
            self.Z_He1 = V * Ztr_He * Zelec_He1
            
            # 2 Ionized Helium 
            Zelec_He2 = np.exp( - (c.xhe1+c.xhe2) / ( c.kb * T))
            self.Z_He2 = V * Ztr_He * Zelec_He2
            
            # Electron
            Ztr_e = (2 * np.pi * c.me * c.kb * T)**(3/2) / c.h**3
            Zspin_e = 2 
            self.Z_e = V * Ztr_e * Zspin_e


    par = partition(T,Vol)
    K_dis = par.Z_H**2 / (par.Z_H2 * Vol)
    K_ion = par.Z_Hion * par.Z_e / (par.Z_H * Vol)
    # logK_ion = np.log10(K_ion)
    K_He1 = par.Z_He1 * par.Z_e / (par.Z_He * Vol)
    K_He2 = par.Z_He2 * par.Z_e / (par.Z_He1 * Vol)
    # logK_He2 = np.log10(K_He2)
    nH = Den * 0.7 / c.mh
    nHe = Den * 0.28 / c.mhe

    def chemical_eq(ne, i):
        oros1 = 2 * ne**2 * nH[i] * K_ion[i] / (np.sqrt( (ne + K_ion[i])**2 + 8 * nH[i] * ne**2 / K_dis[i]) + ne + K_ion[i])
        oros2 = (K_He1[i] * ne + 2 * K_He1[i] * K_He2[i]) * nHe[i] * ne**2 / (ne**2 + K_He1[i] * ne + K_He1[i] * K_He2[i])
        oros3 = -ne**3
        return oros1 + oros2 + oros3

    ne_sol = np.zeros(len(nH))
    for i in tqdm(range(len(nH))):
        ne_sol[i] = bisection(chemical_eq, 1e-15, 1e50, i)

    inv_ne_sol = 1/ne_sol
    #%% With n_e, get ionization fractions

    # He
    oros = 1 + K_He1 * inv_ne_sol + K_He1 * K_He2*inv_ne_sol**2
    nHe_sol = np.divide(nHe, oros) # Eq 83

    # He+
    n_He1_sol = nHe_sol * K_He1 * inv_ne_sol # Eq 77
    xHe1 = n_He1_sol / (n_He1_sol + nHe_sol)

    # He++
    n_He2_sol = nHe_sol * K_He2 * inv_ne_sol # Eq 78
    xHe2 = n_He2_sol / (n_He2_sol + nHe_sol)

    # H
    orosH = ne_sol - nHe_sol * inv_ne_sol * K_He1 - \
        2 * nHe_sol - inv_ne_sol**2 * K_He1 * K_He2
    alpha = 2 / K_dis
    beta = 1 + K_ion/ne_sol
    gamma = -nH
    delta = beta**2 - 4 * alpha * gamma
    nH_sol = (-beta + np.sqrt(delta)) / (2*alpha)
    # nH_sol2 = (-beta - np.sqrt(delta)) / (2*alpha)
    # nH_sol = np.divide(ne_sol, K_ion) * orosH  # Eq 84

    # H+
    n_Hion_sol = nH_sol * K_ion * inv_ne_sol # Eq 76
    xH = n_Hion_sol / (n_Hion_sol + nH_sol)
    #xH = np.nan_to_num(xH, nan=1)
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
        #%% Plot stream sections

if alice:
    np.savetxt(f'{pre}tde_comparison/data/ion{sim}.txt', transition_slices)

if plot:
    plt.ioff()
    for check in tqdm(range(52, 53)):#range(ray_no)):#ray_no) ):
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
        #plt.savefig(f'/home/konstantinos/ionfig/{figno}.png', dpi = 300)
        #plt.close()
