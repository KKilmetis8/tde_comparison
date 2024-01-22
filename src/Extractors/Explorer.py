# -*- coding: utf-8 -*-
"""
Extracts data from h5 files
"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

import numpy as np
import h5py
from datetime import datetime
snapshot881 = "6/881/snap_881.h5"
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
        # h5printR(h, '  ')
        a = h['Box']
        print(a[5]/1.4)
        
h5print(snapshot881)

    
    
    
    
    
    
    
    
    
    
    
    
    