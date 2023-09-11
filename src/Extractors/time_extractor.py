# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 21:51:56 2022

@author: Konstantinos


"""
#%% Imports

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.figsize'] = [10.0, 5.0]
plt.rcParams['figure.dpi'] = 300
from datetime import datetime
import h5py

#%% Extract Energy
snapshot881 = "547/snap_547.h5"
snapshot881 = "881/snap_881.h5"
snapshot820 = "820/snap_820.h5"
snapshot1008 = '1008/snap_1008.h5'
#%% Get Energies

## File structure is
# box, cycle, time, mpi, rank0 ... rank99.
# This iterates over all the ranks

def days_since_distruption(filename):
    '''
    Loads the file, extracts specific kinetic and potential energies 
    
    Parameters
    ----------
    f : str, 
        hdf5 file name. Contains the data
    
    Returns
    -------
    days: float, days since the distruption begun.
    
    '''
    # Timing start
    start_time = datetime.now()
    # Read File
    f = h5py.File(filename, "r")
    G = 6.6743e-11 # SI
    Msol = 1.98847e30 # kg
    Rsol = 6.957e8 # m
    t = np.sqrt(Rsol**3 / (Msol*G )) # Follows from G=1
    Mbh = 1e6 # * Msol
    time = np.array(f['Time'])
    days = time.sum()*t / (24*60*60)
    return days



#%% 
def linear_fit_days(x):
    '''
    Converts from snapshot number to the more 
    intuitive days since distruption metric. 
    
    Uses a linear fit from snapshots 820, 881, and,
    1008 and thus could prove to not be 100% accurate.
    
    Parameters
    ----------
    x : int,
        Snapshot number to convert from.

    Returns
    -------
    y : int,
        Days since distruption.

    '''
    days820 = days_since_distruption(snapshot820)
    days881 = days_since_distruption(snapshot881)
    days1008 = days_since_distruption(snapshot1008)
    snaps = [820, 881, 1008]
    days = [days820, days881, days1008]
    time_fit = np.polyfit(snaps, days, deg=1)
    y = time_fit[0]*x + time_fit[1]
    return int(y)
#%%
if __name__ == '__main__':
    snaps = [820, 881, 1008]
    days820 = days_since_distruption(snapshot820)
    days881 = days_since_distruption(snapshot881)
    days1008 = days_since_distruption(snapshot1008)
    days = [days820, days881, days1008]
    testspace = np.linspace(800,1010)
    # Plot
    plt.rcParams['figure.figsize'] = [4.0, 4.0]
    plt.plot(snaps, days, 'o-', label='real', color='navy')
    plt.plot(testspace, linear_fit_days(testspace), label='fit', color='maroon')
    plt.grid()
    plt.legend()
        
    
    
    
    
    
    
    
    
    
