#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 19:32:10 2025

@author: konstantinos
"""

import pandas as pd
import numpy as np

df4 = pd.read_csv(f'data/photosphere/richex_photocolor4.csv', 
                 sep = ',',
                 comment = '#', header = None)
df5 = pd.read_csv(f'data/photosphere/richex_photocolor5.csv', 
                 sep = ',',
                 comment = '#', header = None)
df6 = pd.read_csv(f'data/photosphere/richex_photocolor6.csv', 
                 sep = ',',
                 comment = '#', header = None)

seeked_times = [0.62, 1.02, 1.42]
for seeked_time in seeked_times:
    for i, df in zip([4,5,6], [df4, df5, df6]):
        idx = np.argmin(np.abs(seeked_time - df.iloc[:,1]))
        snap = df.iloc[idx, 0]
        print(f'{seeked_time} tfb in {i}, snap is {snap}')