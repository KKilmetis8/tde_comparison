#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 17:38:24 2024
@author: konstantinos
"""

import numpy as np
import h5py
from src.Utilities.parser import parse

## File structure is
# box, cycle, time, mpi, rank0 ... rank99.
# This iterates over all the ranks
def extractor(filename):
    # Read File
    f = h5py.File(filename, "r")
    box = np.zeros(6)
    for i in range(len(box)):
        box[i] = f['Box'][i]

    # Close the file
    f.close()
    return box
#%% Time extractor

#%% Do it    
def main():
    args = parse()
    simname = args.name

    fixes = np.arange(args.first, args.last + 1)
    realpre = '/data1/kilmetisk/TDE/'
    for fix in fixes:
        snapshot = f'{realpre}{simname}/snap_{fix}/snap_{fix}.h5'
        pre = f'{realpre}{simname}/snap_{fix}/'
        suf = f'_{fix}'
        
        try:
            box = extractor(snapshot)
        except FileNotFoundError:
            continue
        print('Did ', fix)
        #%% Save to another file.
        np.save(pre + 'box' + suf, box)
            
if __name__ == "__main__":
    main()
