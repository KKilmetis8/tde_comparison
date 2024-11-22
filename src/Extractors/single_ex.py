#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 17:38:24 2024

@author: konstantinos
"""

import numpy as np
import h5py
import os
import argparse

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

    # Iterate over ranks
    for key in keys:
        if key in not_ranks:
            # Skip whatever is not a mpi rank
            continue
        else:
            # For some reason, having the collumns into variables is way faster.
            x_data = f[key]['Dissipation']
            for i in range(len(x_data)):
                X.append(x_data[i])

    # Close the file
    f.close()
    return X
#%% Time extractor

def time_extractor(mbh, snapno, mass, radius, pre):
    snap = f'{pre}/snap_{snapno}.h5'
    f = h5py.File(snap, "r")
    G = 6.6743e-11 # SI 
    Msol = 1.98847e30 # kg
    Rsol = 6.957e8 # m
    t = np.sqrt(Rsol**3 / (Msol*G )) # Follows from G=1
    
    #mbh = 10**m
    time = np.array(f['Time'])
    days = time.sum()*t / (24*60*60)
    tfb = 40 * np.power( mbh/1e6, 1/2) * np.power(mass,-1) * np.power(radius, 3/2)
    np.savetxt(f'{pre}/tbytfb_{snapno}.txt',[days/tfb])

#%% Do it    

def main():
    # Parse the command-line arguments
    from src.Utilities.parser import parse
    args = parse()
    simname = args.name
    m = args.mass
    r = args.radius
    mbh = float(args.blackhole)
    fixes = np.arange(args.first, args.last + 1)
    realpre = '/data1/kilmetisk/TDE/'
    for fix in fixes:
        star = 'half'
        snapshot = f'{realpre}{simname}/snap_{fix}/snap_{fix}.h5'
        pre = f'{realpre}{simname}/snap_{fix}/'
        suf = f'_{fix}'
        
        try:
            Diss = extractor(snapshot)
        except FileNotFoundError:
            continue
        #%% Save to another file.
        np.save(pre + 'Diss' + suf, Diss)   
        
        #%% Do time
        # time_extractor(mbh, fix, m,  r, pre)
    
            
if __name__ == "__main__":
    main()

