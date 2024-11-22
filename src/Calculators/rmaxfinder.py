#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 11:19:56 2024

@author: konstantinos
"""
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import src.Utilities.prelude as c
from src.Utilities.parser import parse
#%% Cross dot -----------------------------------------------------------------
pre = 6
snap = 351
def rfinder(pre, snap, N_ray=5000):
    observers_xyz = hp.pix2vec(c.NSIDE, range(192))
    # Line 17, * is matrix multiplication, ' is .T
    observers_xyz = np.array(observers_xyz).T
    cross_dot = np.matmul(observers_xyz,  observers_xyz.T)
    cross_dot[cross_dot<0] = 0
    cross_dot *= 4/192
    box = np.load(f'{pre}/{snap}/box_{snap}.npy')
        
    for i in range(192):
        # Progress 
        # if i % 10 == 0:
        #     print('Eladython Ray no:', i)
        # print(i)
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
            
        return np.logspace( -0.25, np.log10(rmax), N_ray)
