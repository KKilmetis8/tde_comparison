#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 16:59:57 2023

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
plt.rcParams['figure.figsize'] = [10 , 6]
plt.rcParams['axes.facecolor']= 	'whitesmoke'

# Data load
pre = 'data/red/alicered'

red6 = np.loadtxt(f'{pre}6fid.txt') 
red4 = np.loadtxt(f'{pre}4fid.txt')
days6 = np.loadtxt(f'{pre}6fid_days.txt')
days4 = np.loadtxt(f'{pre}4fid_days.txt')
# days4 = [0.505,0.515,0.525,0.5325,0.5425,0.5525,0.56,0.57,0.58,0.59,0.5975,0.6075,0.6175,0.625,0.635,0.645,0.6525,0.6625,0.6725,0.68,0.69,0.7,0.71,0.7175,0.7275,0.7375,0.745,0.755,0.765,0.7725,0.7825,0.7925,0.8,0.81,0.82,0.83,0.8375,0.8475,0.8575,0.865,0.875,0.885,0.8925,0.9025,0.9125,0.92,0.93,0.94,0.95,0.9575,0.9675,0.9775,0.985,0.995,1.005,1.0125,1.0225,1.0325,1.04,1.05,1.06,1.0675,1.0775,1.0875,1.0975,1.105,1.115,1.125,1.1325,1.1425,1.1525,1.16,1.17,1.18,1.1875,1.1975,1.2075,1.2175,1.225,1.235,1.245,1.2525,1.2625,1.2725,1.28,1.29,1.3,1.3075,1.3175,1.3275,1.335,1.345,1.355,1.365,1.375,1.3825,1.3925,1.4025,1.41,1.42,1.43,1.4375,1.4475,1.4575,1.4675,1.475,1.485,1.495,1.5025,1.5125,1.5225,1.53,1.54,1.55,1.5575,1.5675,1.5775,1.585,1.5975,1.6075,1.615,1.625,1.635,1.6425,1.6525,1.6625,1.6725,1.68,1.69,1.7,1.7075,1.7175,1.7275,1.735,1.745,1.755,1.7625,1.7725,1.7825,1.7925,1.8,1.81,1.82,1.8275,1.8375]
# days4_CHR = [0.7825,0.79,0.8,0.81,0.8175,0.8275,0.8375,0.845,0.855,0.865,0.875,0.8825,0.8925,0.9025,0.91,0.92,0.93,0.9375,0.9475,0.9575,0.965,0.975,0.985,0.995,1.0025,1.0125,1.0225,1.03,1.04,1.05,1.0575,1.0675,1.0775,1.085,1.095,1.105,1.115,1.1225,1.1325,1.1425,1.15,1.16,1.17,1.1775,1.1875,1.1975,1.205,1.215,1.225,1.235,1.2425,1.2525,1.2625,1.27,1.28,1.29,1.2975,1.3075,1.3175,1.325,1.335,1.345]
# red4_CHR = np.loadtxt(pre + '4-S60ComptonHires.txt')

plt.plot(days6, red6, c = 'tab:red', label = '6')
#plt.plot(red6[0], red6[1], c = 'tab:red', label = '6')
plt.plot(days4, red4, c = 'g', label = '4')#- fiducial')
#plt.plot(days4_CHR, red4_CHR, c = 'tab:green', label = '4 - Compton HiRes')

Ledd4 = 1.26e38 * 1e4
Ledd6 = 1.26e38 * 1e6
# plt.axhline(Ledd4, c = 'k', linestyle = 'dotted')
# plt.axhline(Ledd6, c = 'k', linestyle = 'dotted')
# plt.text(0.45, Ledd4 * 1.2, 'L$_{edd}$ $10^4 M_{\odot}$', fontsize = 16)
# plt.text(0.45, Ledd6 * 1.2, 'L$_{edd}$ $10^6 M_{\odot}$', fontsize = 16)

plt.yscale('log')
plt.xlabel('Time [t/t$_{FB}$]')
plt.ylabel('Luminosity [erg/s]')
plt.title('Flux Limited Diffusion')
plt.grid()
plt.legend(loc = 'best')
plt.savefig('Plightcurves46.png')
#plt.savefig('Final plot/lightcurves46.png')
plt.show()
