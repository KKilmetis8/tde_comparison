#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test if the bolometric luminosity from the single observers gives you the same value as red

@author: paola 

"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

from src.Utilities.isalice import isalice
alice, plot = isalice()

# Vanilla imports
import numpy as np
import matplotlib.pyplot as plt
import h5py

plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [5 , 4]
pre = '/home/s3745597/data1/TDE/tde_comparison'

m = 6
snap = 881
nL_tilde_n = np.loadtxt(f'data/blue/nLn_single_m{m}_{snap}_all.txt')
nL_tilde_n_new = np.zeros(len(nL_tilde_n[1]))
for i in range(len(nL_tilde_n[1])):
    for iobs in range(len(nL_tilde_n_new)):
        nL_tilde_n_new[i] += nL_tilde_n[iobs][i]
#nL_tilde_n_new /= 192
x_array = np.loadtxt(f'data/blue/spectrafreq_m{m}.txt')
n_array = np.power(10, x_array)

# Ltot = 0 
# for i in range(len(nL_tilde_n)):
#     xLx =  n_array * nL_tilde_n[i]
#     L = np.trapz(xLx, x_array) 
#     L *= np.log(10)
#     Ltot += L
# Ltot /= len(nL_tilde_n)

# print(Ltot)

L = np.trapz(n_array * nL_tilde_n_new, x_array) 
L *= np.log(10)
L /= len(nL_tilde_n)
print(L)

with open(f'data/test.txt', 'a') as file:  
    file.write(' '.join(map(str, nL_tilde_n_new)) + '\n')
    file.close()

with h5py.File(f'data/elad/data_881.mat', 'r') as f:
    print(f.keys())