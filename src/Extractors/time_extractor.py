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
# snapshot233= "/Users/paolamartire/tde_comparison/4/233/snap_full_233.h5"
# snapshot254= "/Users/paolamartire/tde_comparison/4/254/snap_full_254.h5"
# snapshot263 = "/Users/paolamartire/tde_comparison/4/263/snap_full_263.h5"
# snapshot277 = "/Users/paolamartire/tde_comparison/4/277/snap_full_277.h5"
# snapshot269 = '5/269/snap_269.h5'
# snapshot844 = '6/844/snap_844.h5'
# snapshot308 = '5/308/snap_308.h5'

#%% Get Energies

## File structure is
# box, cycle, time, mpi, rank0 ... rank99.
# This iterates over all the ranks

def days_since_distruption(filename, mbh, mstar, rstar):
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

    time = np.array(f['Time'])
    days = time.sum()*t / (24*60*60)
    tfb = 40 * np.power( mbh/1e6, 1/2) * np.power(mstar,-1) * np.power(rstar, 3/2)
    # print('days/tfb', days/tfb)
    # print('days', days)
    return days/tfb

#%%

def time_extractor(m, star, snapno, mass, radius):
    snap = f'{m}{star}/{snapno}/snap_full_{snapno}.h5'
    tbytfb = days_since_distruption(snap,10**m,mass,radius)
    np.savetxt(f'{m}{star}/{snapno}/tbytfb_{snapno}.txt',[tbytfb])
    
if __name__ == '__main__':
    # snaps = [820, 881, 254]
    # days820 = days_since_distruption(snapshot820)
    # days233= days_since_distruption(snapshot881)
    # days254 = days_since_distruption(snapshot254)
    # days = [days820, days881, days254]
    # testspace = np.linspace(800,1010)
    # # Plot
    # plt.rcParams['figure.figsize'] = [4.0, 4.0]
    # plt.plot(snaps, days, 'o-', label='real', color='navy')
    # plt.plot(testspace, linear_fit_days(testspace), label='fit', color='maroon')
    # plt.grid()
    # plt.legend()
    # days322 = linear_fit_days(322)
    #print(days322)
    m = 4
    snapno = 200
    snap = f'{m}/{snapno}/snap_full_{snapno}.h5'
    tbytfb = days_since_distruption(snap,10**m,1.0,1.0)
    np.savetxt(f'{m}/{snapno}/tbytfb_{snapno}.txt',[tbytfb])
        
    
    
    
    
    
    
    
    
    
