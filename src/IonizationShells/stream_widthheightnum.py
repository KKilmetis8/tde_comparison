#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 09:37:57 2024

@author: konstantinos
"""
# Vanilla
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle 

# Chocolate
import src.Utilities.prelude as c

# NOTE: TRACK THE NUMBER OF CELLS IN THE STREAM
#%% Data Import
def stream_getter(fix = 199, sim = '4half', thetanum = 100):
    import os
    print(os.system(f'ls {sim}/{fix}/'))
    print(f'\n Getting stream for fix {fix} from sim {sim}')
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
    ray_no = thetanum
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
        
        # t_coord = np.dot( [ den_max_point['x'], den_max_point['y']], that)
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
                n_coord = np.dot([cell['x'], cell['y']], nhat)
                stream[i].append((j, cell['z'], n_coord, cell['x'], cell['y'], 
                                  cell['ion1'], cell['ion2'], cell['ion3'],  
                                  cell['T'], cell['den']))
        
    with open(f'{sim}{fix}stream.pkl', 'wb') as f:
        pickle.dump(stream, f)
    with open(f'{sim}{fix}denmax.pkl', 'wb') as f:
        pickle.dump(density_maxima, f)

    return (stream, density_maxima)
#%% Do it
thetas =  np.linspace(-np.pi, np.pi, num = 100)
stream_getter(199, '4half', 100)
stream_getter(199, '4halfHR', 100)
#stream_getter(169, '4halfSHR', 100)

#%% Load it back
