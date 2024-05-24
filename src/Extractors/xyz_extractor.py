#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 17:06:56 2023

@author: konstantinos
"""
import os
import numpy as np
import h5py
from datetime import datetime

from src.Extractors.time_extractor import time_extractor
import src.Utilities.selectors as s
from src.Utilities.isalice import isalice

alice, _ = isalice()

#%% Get Densities

## File structure is
# box, cycle, time, mpi, rank0 ... rank99.
# This iterates over all the ranks



def extractor(filename, m):
    '''
    Loads the file, extracts X,Y,Z and Density. 
    
    Parameters
    ----------
    f : str, 
        hdf5 file name. Contains the data
    
    Returns
    -------
    X : np.array, float64
        X - coordinate
    Y : np.array, float64
        Y - coordinate.
    Z : np.array, float64
        Z - coordinate.
    Den : np.array, float64
        Density.
    
    '''
    # Timing start
    start_time = datetime.now()
    # Read File
    f = h5py.File(filename, "r")
    # HDF5 are dicts, get the keys.
    keys = f.keys() 
    # List to store the length of each rank
    lengths = []
    # List with keys that don't hold relevant data
    not_ranks = ['Box', 'Cycle', 'Time', 'mpi']
    
    for key in keys:
        if key in not_ranks:
            # Skip whatever is not a mpi rank
            continue
        else:
            # Store the length of the dataset
            lengths.append(len(f[key]['X']))
    
    # Use lists for clarity
    X = []
    Y = []
    Z = []
    Den = []
    Vx = []
    Vy = []
    Vz = []
    Vol = []
    Mass = []
    IE = []
    Rad = []
    T = []
    P = []
    star = []
    
    # Iterate over ranks
    for key in keys:
        if key in not_ranks:
            # Skip whatever is not a mpi rank
            continue
        else:
            # Sanity Check
            print(key)
            # Timing
            end_time = datetime.now()
            print('Duration: {}'.format(end_time - start_time))
            # For some reason, having the collumns into variables is way faster.
            x_data = f[key]['CMx']
            y_data = f[key]['CMy']
            z_data = f[key]['CMz']
            den_data = f[key]['Density']
            star_data = f[key]['tracers']['Star']
            vx_data = f[key]['Vx']
            vy_data = f[key]['Vy']
            vz_data = f[key]['Vz']
            vol_data = f[key]['Volume']
            
            ie_data = f[key]['InternalEnergy']
            if m == 6:
                rad_data = f[key]['tracers']['ZRadEnergy'] 
            else:
                rad_data = f[key]['Erad']  #
            T_data = f[key]['Temperature']
            P_data = f[key]['Pressure']
            for i in range(len(x_data)):
                X.append(x_data[i])
                Y.append(y_data[i])
                Z.append(z_data[i])
                Den.append(den_data[i])
                Vx.append(vx_data[i])
                Vy.append(vy_data[i])
                Vz.append(vz_data[i])
                Vol.append(vol_data[i])
                IE.append(ie_data[i])
                Rad.append(rad_data[i])
                Mass.append(vol_data[i] * den_data[i])
                T.append(T_data[i])
                P.append(P_data[i])
                star.append(star_data[i])


    # Close the file
    f.close()
    return X, Y, Z, Den, Vx, Vy, Vz, Vol, Mass, IE, Rad, T, P, star
#%%
# Change the current working directory
m = 5
mstar = 0.5
if mstar == 0.5:
    star = 'half'
else:
    star = ''
rstar = 0.47
check = 'fid'
snapshots, _ = s.select_snap(m, mstar, rstar, check)

for fix in snapshots:
    if alice:
        pre_file = f'/home/s3745597/data1/TDE/{m}{star}-{check}/snap_{fix}'
    else:
        pre_file = f'{m}/{fix}'

    if os.path.isfile(pre_file):
        snapshot = f'{pre_file}/snap_{fix}.h5'
        pre = f'{pre_file}/'
        suf = f'_{fix}'

        X, Y, Z, Den, Vx, Vy, Vz, Vol, Mass, IE, Rad, T, P, Star = extractor(snapshot, m)
        
        #%% Save to another file.
        np.save(pre + 'CMx' + suf, X)   
        np.save(pre + 'CMy' + suf, Y) 
        np.save(pre + 'CMz' + suf, Z) 
        np.save(pre + 'Den' + suf, Den)
        np.save(pre + 'Vx' + suf, Vx)   
        np.save(pre + 'Vy' + suf, Vy) 
        np.save(pre + 'Vz' + suf, Vz)
        np.save(pre + 'Vol' + suf, Vol)
        np.save(pre + 'Mass' + suf, Mass)   
        np.save(pre + 'IE' + suf, IE) 
        np.save(pre + 'Rad' + suf, Rad)
        np.save(pre + 'T' + suf, T)
        np.save(pre + 'P' + suf, P) 
        np.save(pre + 'Star' + suf, Star)
        
        #%% Do time
        #time_extractor(m, star, fix, 0.5, 0.47)
    
    
            
