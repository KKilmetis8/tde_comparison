# -*- coding: utf-8 -*-
"""
Extracts data from h5 files
"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

import numpy as np
import h5py
from datetime import datetime
import src.Utilities.prelude as c
snapshot881 = "4/199/snap_full_199.h5"
# f = h5py.File(snapshot881, "r")
#%% Explores the structure of the hdf5


def h5printR(item, leading = ''):
    for key in item:
        if isinstance(item[key], h5py.Dataset):
            print(leading + key + ': ' + str(item[key].shape))
        else:
            print(leading + key)
            h5printR(item[key], leading + '  ')

# Print structure of a `.h5` file            
def h5print(filename):
    with h5py.File(filename, 'r') as h:
        print(filename)
        h5printR(h, '  ')
        sec_to_day = 60*60*24
        a = np.array([h['Time']])[0] * c.t / sec_to_day
        m = 5 
        mstar = 0.5 # Msol
        rstar = 0.47 # Rsol
        t_fb = 40 * np.sqrt(10**(m - 6)) * np.power(mstar,-1) * np.power(rstar, 3/2)
        print(t_fb)
        print(a / t_fb)
        
h5print(snapshot881)

    
    
    
    
    
    
    
    
    
    
    
    
    