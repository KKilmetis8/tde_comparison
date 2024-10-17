#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 00:15:08 2024

@author: konstantinos
"""
# Vanilla
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

age = 1500
ext = True
if ext:
    pre = '/media/konstantinos/Dunkey/mesadata/jups02'
else:
    pre = 'data'
df = pd.read_csv(f'{pre}/specific_ages/{age}.txt', header = None, 
                              delimiter = '\s', names = ['Planet', 'Profile'])

for planet, profile_no in tqdm(zip(df['Planet'], df['Profile'])):
    profile = f'{pre}/{planet}/profile{profile_no}.data'
    new_name = planet + '_' + f'profile{profile_no}.data'
    destination = f'data/{age}profile/' + new_name
    os.system(f'cp {profile} {destination}')
    
