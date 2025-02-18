#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 12:14:15 2025

@author: konstantinos
"""

import numpy as np

def local_loader(m, fix, what):
    'Loads extracted .npy data from my local machine'
    if what == 'orbital':
        X = np.load(f'{m}/{fix}/CMx_{fix}.npy')
        Y = np.load(f'{m}/{fix}/CMy_{fix}.npy')
        Z = np.load(f'{m}/{fix}/CMz_{fix}.npy')
        VX = np.load(f'{m}/{fix}/Vx_{fix}.npy')
        VY = np.load(f'{m}/{fix}/Vy_{fix}.npy')
        VZ = np.load(f'{m}/{fix}/Vz_{fix}.npy')
        time = np.loadtxt(f'{m}/{fix}/tbytfb_{fix}.txt')
        return X, Y, Z, VX, VY, VZ, time
    if what == 'orbital+mass':
        X = np.load(f'{m}/{fix}/CMx_{fix}.npy')
        Y = np.load(f'{m}/{fix}/CMy_{fix}.npy')
        Z = np.load(f'{m}/{fix}/CMz_{fix}.npy')
        VX = np.load(f'{m}/{fix}/Vx_{fix}.npy')
        VY = np.load(f'{m}/{fix}/Vy_{fix}.npy')
        VZ = np.load(f'{m}/{fix}/Vz_{fix}.npy')
        Den = np.load(f'{m}/{fix}/Den_{fix}.npy')
        Vol = np.load(f'{m}/{fix}/Vol_{fix}.npy')
        Mass = Den * Vol
        time = np.loadtxt(f'{m}/{fix}/tbytfb_{fix}.txt')
        return X, Y, Z, VX, VY, VZ, Mass, time
    if what == 'thermodynamics':
        X = np.load(f'{m}/{fix}/CMx_{fix}.npy')
        Y = np.load(f'{m}/{fix}/CMy_{fix}.npy')
        Z = np.load(f'{m}/{fix}/CMz_{fix}.npy')
        Den = np.load(f'{m}/{fix}/Den_{fix}.npy')
        T = np.load(f'{m}/{fix}/T_{fix}.npy')
        Rad = np.load(f'{m}/{fix}/Rad_{fix}.npy')
        Vol = np.load(f'{m}/{fix}/Vol_{fix}.npy')
        box = np.load(f'{m}/{fix}/box_{fix}.npy')
        day = np.loadtxt(f'{m}/{fix}/tbytfb_{fix}.txt')
        return X, Y, Z, Den, T, Rad, Vol, box, day
    
def boxer(i, observers_xyz, box):
    ''' Gets you the maximum box size, for a given solid angle, 
        observers_xyz comes from hp.pix2vec'''
        
    mu_x = observers_xyz[i][0]
    mu_y = observers_xyz[i][1]
    mu_z = observers_xyz[i][2]

    # Box is for dynamic ray making
    if mu_x < 0:
        rmax = box[0] / mu_x
    else:
        rmax = box[3] / mu_x
    if mu_y < 0:
        rmax = min(rmax, box[1] / mu_y)
    else:
        rmax = min(rmax, box[4] / mu_y)

    if mu_z < 0:
        rmax = min(rmax, box[2] / mu_z)
    else:
        rmax = min(rmax, box[5] / mu_z)
    return rmax

def alice_loader(sim, fix, what):
    'Loads extracted .npy data from my local machine'
    realpre = '/home/kilmetisk/data1/TDE/'
    pre = f'{realpre}{sim}/snap_'
    if what == 'orbital':
        X = np.load(f'{pre}{fix}/CMx_{fix}.npy')
        Y = np.load(f'{pre}/{fix}/CMy_{fix}.npy')
        Z = np.load(f'{pre}/{fix}/CMz_{fix}.npy')
        VX = np.load(f'{pre}/{fix}/Vx_{fix}.npy')
        VY = np.load(f'{pre}/{fix}/Vy_{fix}.npy')
        VZ = np.load(f'{pre}/{fix}/Vz_{fix}.npy')
        time = np.loadtxt(f'{pre}/{fix}/tbytfb_{fix}.txt')
        return X, Y, Z, VX, VY, VZ, time
    if what == 'orbital+mass':
        X = np.load(f'{pre}/{fix}/CMx_{fix}.npy')
        Y = np.load(f'{pre}/{fix}/CMy_{fix}.npy')
        Z = np.load(f'{pre}/{fix}/CMz_{fix}.npy')
        VX = np.load(f'{pre}/{fix}/Vx_{fix}.npy')
        VY = np.load(f'{pre}/{fix}/Vy_{fix}.npy')
        VZ = np.load(f'{pre}/{fix}/Vz_{fix}.npy')
        Den = np.load(f'{pre}/{fix}/Den_{fix}.npy')
        Vol = np.load(f'{pre}/{fix}/Vol_{fix}.npy')
        Mass = Den * Vol
        time = np.loadtxt(f'{pre}/{fix}/tbytfb_{fix}.txt')
        return X, Y, Z, VX, VY, VZ, Mass, time
    if what == 'thermodynamics':
        X = np.load(f'{pre}/{fix}/CMx_{fix}.npy')
        Y = np.load(f'{pre}/{fix}/CMy_{fix}.npy')
        Z = np.load(f'{pre}/{fix}/CMz_{fix}.npy')
        Den = np.load(f'{pre}/{fix}/Den_{fix}.npy')
        T = np.load(f'{pre}/{fix}/T_{fix}.npy')
        Rad = np.load(f'{pre}/{fix}/Rad_{fix}.npy')
        Vol = np.load(f'{pre}/{fix}/Vol_{fix}.npy')
        box = np.load(f'{pre}/{fix}/box_{fix}.npy')
        day = np.loadtxt(f'{pre}/{fix}/tbytfb_{fix}.txt')
        return X, Y, Z, Den, T, Rad, Vol, box, day