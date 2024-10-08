#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 15:04:06 2024

@author: konstantinos
"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

import src.Utilities.prelude
from src.Utilities.isalice import isalice
alice, plot = isalice()
import numpy as np
import os
from src.Extractors.time_extractor import days_since_distruption

def select_prefix(m, check, mstar):
    if mstar == 0.5:
        star = 'half'
    else:
        star = ''
    if alice:
        prealice = f'/home/s3745597/data1/TDE/{m}{star}-{check}/snap_'
        pre = prealice
    else:
        pre = f'{m}/'
    return pre

def select_snap(m, mstar, rstar, check, time = False):
    Mbh = 10**m
    pre = select_prefix(m, check, mstar)
    days = []
    if alice:
        if m == 6 and check == 'fid':
            snapshots = np.arange(683, 1008 + 1, step = 1)
        if m == 4 and check == 'fid':
            snapshots = [293,322] #np.arange(110, 322 + 1) 
        if m == 4 and check == 'S60ComptonHires':
            snapshots = np.arange(210, 278 + 1)
        if m == 5 and check == 'fid':
            snapshots = np.arange(100,365+1) 
        # select just the ones that actually exist
        snapshots = [snap for snap in snapshots if os.path.exists(f'{pre}{snap}/snap_{snap}.h5')]
    else:
        if m == 4 and check == 'fid':
            snapshots = [293,322] #, 254, 263, 277 , 293, 308, 322]
        if m == 4 and check == 'S60ComptonHires':
            snapshots = [234] 
        if m == 5:
            snapshots = [308]
        if m == 6 and check == 'fid': 
            snapshots = [844,1008] # [844, 882, 925, 950]#, 1008] 
    for snap in snapshots:
        snap = str(snap) 
        if time:
            t_by_tfb = np.round(days_since_distruption( pre +
                        snap + '/snap_' + snap + '.h5', Mbh, mstar, rstar), 1)
            days.append(t_by_tfb)
            return snapshots, days
        else:
            return snapshots

# Select opacity
def select_opacity(m):
    if m==6:
        return 'cloudy'
    else:
        return 'LTE'