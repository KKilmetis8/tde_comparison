# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 18:02:14 2023

@author: Konstantinos
"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

import numpy as np
import h5py
from datetime import datetime
from src.Extractors.time_extractor import days_since_distruption
#%% Extractor

## File structure is
# box, cycle, time, mpi, rank0 ... rank99.
# This iterates over all the ranks

def extractor(filename):
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
    # List with keys that don't hold relevant data
    not_ranks = ['Box', 'Cycle', 'Time', 'mpi']
    
    # Use lists for clarity
    T = []
    Z = []
    # Iterate over ranks
    for key in keys:
        if key in not_ranks:
            # Skip whatever is not a mpi rank
            continue
        else:
            # Sanity Check & Timing
            printing_ranks = ['rank1','rank2','rank3','rank4','rank5','rank6',
                              'rank7','rank8','rank9']
            end_time = datetime.now()
            if key in printing_ranks:
                print(key)
                print('Duration: {}'.format(end_time - start_time))
                
            # For some reason, having the collumns into variables is way faster.
            T_data = f[key]['Volume']
            Z_data = f[key]['tracers']['Star']
            for i in range(len(T_data)):
                T.append(T_data[i])
                Z.append(Z_data[i])
    # Close the file
    f.close()
    return T, Z

#%% Doing the thing
fixes = [351]
for fix in fixes:
    m = 6
    snapshot = f'{m}/{fix}/snap_full_{fix}.h5'
    _, Z = extractor(snapshot)   
    # Save to another file.
    np.save(f'{m}/{fix}/Star_{fix}', Z)


    
    
    
    
    
    
    
    
    
    
    
    
    
    