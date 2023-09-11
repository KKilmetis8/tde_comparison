#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 18:29:30 2023

@author: konstantinos
"""
import sys
sys.path.insert(0, '/data1/s3745597/TDE/')

import os
import subprocess
import numpy as np
from src.Extractors.time_extractor import days_since_distruption

folders = ['tde_data2', 'tde_data2s30']
txts = []
wd = os.getcwd()
for i in range(len(folders)):
    os.chdir(folders[i])
    ls = subprocess.check_output(['ls' ,'-l'], text=True)
    snap_num = ls.count('\n') - 1 # Count lines, remove one for the 1st
    fixes = np.arange(1, snap_num) 
    for fix in fixes:
        fix = str(fix)
        try:
            day = np.round(days_since_distruption('snap_' + fix+'/snap_' + fix + '.h5') , 2)
        except:
            print('Snap ' + fix + ' is missing')
            continue
        txt = fix + ' | ' + str(day)
        print(txt)
    print('-----')
    os.chdir(wd) # Go back to the original directory
# print(txts)
