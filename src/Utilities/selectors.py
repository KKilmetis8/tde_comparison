#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 15:04:06 2024

@author: konstantinos
"""
import src.Utilities.prelude
from src.Utilities.isalice import isalice
alice, plot = isalice()
import numpy as np
from src.Extractors.time_extractor import days_since_distruption

def select_prefix(m, check):
    if alice:
        prealice = '/home/s3745597/data1/TDE/' + str(m) + '-' + check + '/snap_'
        pre = prealice
    else:
        pre = str(m) + '/'
    return pre

def select_snap(m, mstar, rstar, check):
    Mbh = 10**m
    pre = select_prefix(m, check)
    days = []
    if alice:
        if m == 6 and check == 'fid':
            snapshots = np.arange(683, 1008 + 1, step = 1)
        if m == 4 and check == 'fid':
            snapshots = np.arange(110, 322 + 1) #np.arange(110, 322 + 1)
        if m == 4 and check == 'S60ComptonHires':
            snapshots = np.arange(210, 278 + 1)
    else:
        if m == 4 and check == 'fid':
            snapshots = [233] #, 254, 263, 277 , 293, 308, 322]
        if m == 4 and check == 'S60ComptonHires':
            snapshots = [234] 
        if m == 5:
            snapshots = [308]
        if m == 6 and check == 'fid': 
            snapshots = [844,1008] # [844, 882, 925, 950]#, 1008] 
    for snap in snapshots:
        snap = str(snap) 
        t_by_tfb = np.round(days_since_distruption( pre +
                    snap + '/snap_' + snap + '.h5', Mbh, mstar, rstar), 1)
        days.append(t_by_tfb)
    return snapshots, days

# Select opacity
def select_opacity(m):
    if m==6:
        return 'cloudy'
    else:
        return 'LTE'