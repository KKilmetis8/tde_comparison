#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 15:10:55 2023

@author: konstantinos
"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.figsize'] = [5 , 4]
plt.rcParams['axes.facecolor']= 	'whitesmoke'

# Elad Load
mat = scipy.io.loadmat('data/elad.mat')
elad_time = mat['time']
elad_blue = mat['L_bb']
elad_red = mat['L_fld']
# Ours Load
m = 6
x = np.loadtxt('data/L_spectrum_m' + str(m) + '.txt')[0] # x = logν
b = np.loadtxt('data/bluedata_m'+ str(m) + '.txt') 
fld_data = np.loadtxt('data/reddata_m'+ str(m) +'.txt')

# Elad Plot
plt.plot(elad_time[0], np.power(10, elad_red[0]), c = 'r')
plt.plot(elad_time[0], np.power(10, elad_blue[0]), c = 'b')
# Our plot
days = [1, 1.1, 1.3, 1.4]
days40 = np.multiply(days, 40)
plt.plot(days40, b, '--s', c='navy', markersize = 4, alpha = 0.8)
plt.plot(days40, fld_data[1], '--o', c='maroon', markersize = 4, alpha = 0.8)

# # Make Pretty
plt.yscale('log')
plt.grid()
#plt.ylim(3e42, 3e44)
plt.xlim(39, 59)
plt.xlabel('Time [days]')
plt.ylabel('Luminosity [erg/s]')
#plt.savefig('Final plot/Elad_comparison.png')
plt.show()