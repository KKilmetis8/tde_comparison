#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 2023

@author: paola 

"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

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

def select_snap(m, check):
    Mbh = 10**m
    t_fall = 40 * (Mbh/1e6)**(0.5)  # days EMR+20 p13
    pre = select_prefix(m, check)
    days = []
    if alice:
        if m == 6 and check == 'fid':
            snapshots = np.arange(844, 1008 + 1, step = 1)
        if m == 4 and check == 'fid':
            snapshots = np.arange(110, 322 + 1) #np.arange(110, 322 + 1)
        if m == 4 and check == 'S60ComptonHires':
            snapshots = np.arange(210, 278 + 1)
    else:
        if m == 4 and check == 'fid':
            snapshots = [233] #, 254, 263, 277 , 293, 308, 322]
        if m == 6 and check == 'fid':
            snapshots = [844, 881, 925, 950, 1008] 
    for snap in snapshots:
        snap = str(snap) 
        day = np.round(days_since_distruption( pre +
                    snap + '/snap_' + snap + '.h5'), 1)
        t_by_tfb = day / t_fall
        days.append(t_by_tfb)
    return snapshots, days

if __name__ == "__main__":
    m = 4
    check = 'fid'
    save = True 
    snapshots, days =  select_snap(m, check)

    if alice:
        pre_saving = f'/home/s3745597/data1/TDE/tde_comparison/data/days{m}{check}'
        with open(f'{pre_saving}.txt', 'a') as fdays:
            fdays.write(' '.join(map(str, days)) + '\n')
            fdays.close()