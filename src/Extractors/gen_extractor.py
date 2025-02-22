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
                T.append(T_data[i])
                P.append(P_data[i])


    # Close the file
    f.close()
    return X, Y, Z, Den, Vx, Vy, Vz, Vol, IE, Rad, T, P,
#%% Doing the thing
fixes = [350]
for fix in fixes:
    m = 6
    snapshot = f'{m}/{fix}/snap_{fix}.h5'
    X, Y, Z, Den, Vx, Vy, Vz, Vol, IE, Rad, T, P  = extractor(snapshot)   
    # Save to another file.
    pre = f'{m}/{fix}/'
    suf = f'_{fix}'
    np.save(pre + 'CMx' + suf, X)   
    np.save(pre + 'CMy' + suf, Y) 
    np.save(pre + 'CMz' + suf, Z) 
    np.save(pre + 'Den' + suf, Den)
    np.save(pre + 'Vx' + suf, Vx)   
    np.save(pre + 'Vy' + suf, Vy) 
    np.save(pre + 'Vz' + suf, Vz)
    np.save(pre + 'Vol' + suf, Vol)
    # np.save(pre + 'Mass' + suf, Mass)   
    np.save(pre + 'IE' + suf, IE) 
    np.save(pre + 'Rad' + suf, Rad)
    np.save(pre + 'T' + suf, T)
    np.save(pre + 'P' + suf, P)  

#%%
# def time_extractor(mbh, snapno, mass, radius, pre):
#     snap = f'{pre}/snap_{snapno}.h5'
#     f = h5py.File(snap, "r")
#     G = 6.6743e-11 # SI 
#     Msol = 1.98847e30 # kg
#     Rsol = 6.957e8 # m
#     t = np.sqrt(Rsol**3 / (Msol*G )) # Follows from G=1
    
#     #mbh = 10**m
#     time = np.array(f['Time'])
#     days = time.sum()*t / (24*60*60)
#     tfb = 40 * np.power( mbh/1e6, 1/2) * np.power(mass,-1) * np.power(radius, 3/2)
#     np.savetxt(f'{pre}/tbytfb_{snapno}.txt',[days/tfb])

# fix = 50
# mstar = 0.5
# rstar = 0.47
# time_extractor(m, fix, mstar,  rstar, f'{m}/{fix}/')