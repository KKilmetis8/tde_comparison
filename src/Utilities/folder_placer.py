#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 14:41:42 2023

@author: konstantinos
"""

import os
import numpy as np 
#%% Change the working directory
# Print the current working directory
print("Current working directory: {0}".format(os.getcwd()))


# this must be arg parsed
work_dir = input()
snap_num = int(input())
# ------------------------

# Change the current working directory
os.chdir(work_dir+'/')

# Print the current working directory
print("Current working directory: {0}".format(os.getcwd()))
#%% Get the snapshot filenames
filenames = os.listdir() # Get all filenames
wanted_filenames = []
# Get snapshots
for filename in filenames:
    if '.h5' in filename and 'snap' in filename:
        wanted_filenames.append(filename)
snap_num = len(wanted_filenames)
#%% Check for missing files
true_snap_nums = np.arange(0,snap_num+1)
import re
# Sorts like s1, s2, s12, s100
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

wanted_filenames = natural_sort(wanted_filenames)
missing_list = []
for i, filename in enumerate(wanted_filenames):
    filename = filename.replace('snap_','')
    filename = filename.replace('.h5','')
    snap_num = int(filename)
    if snap_num != true_snap_nums[i]:
        print('Missing file:', true_snap_nums[i])
        missing_list.append(true_snap_nums[i])

#%% Put into folders
for filename in wanted_filenames:
    # Makes a folder
    foldername = filename.replace('.h5','')
    os.mkdir(foldername)
    # Puts file in above folder
    os.replace(filename, foldername+'/'+filename)