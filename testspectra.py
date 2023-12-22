#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 9 2023

@author: , paola 


"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

from src.Utilities.isalice import isalice
alice, plot = isalice()

# Vanilla imports
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [5 , 4]
pre = '/home/s3745597/data1/TDE/tde_comparison'

m = 6
nL_tilde_n = np.loadtxt(f'data/blue/nLn_single_m{m}.txt')
x_array = np.loadtxt(f'data/blue/spectrafreq_m{m}.txt')
n_array = np.power(10, x_array)

Ltot = 0 
for i in range(len(nL_tilde_n)):
    xLx =  n_array * nL_tilde_n[i]
    L = np.trapz(xLx, x_array) 
    L *= np.log(10)
    Ltot += L
Ltot /= len(nL_tilde_n)

print(Ltot)