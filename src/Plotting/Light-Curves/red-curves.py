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
pre = 'data/alicered/red'
# 1.184e40 was a nan
days6_before = [0.5,0.5025,0.505,0.508,0.5105,0.513,0.5155,0.518,0.5205,0.523,0.52575,0.52825,0.53075,0.53325,0.536,0.5385,0.541,0.54375,0.54625,0.549,0.5515,0.55425,0.557,0.5595,0.56225,0.56475,0.5675,0.57025,0.57275,0.5755,0.57825,0.581,0.58375,0.5865,0.58925,0.592,0.5945,0.59725,0.6,0.60275,0.6055,0.60825,0.61125,0.614,0.61675,0.6195,0.62225,0.62525,0.628,0.63075,0.63375,0.6365,0.63925,0.64225,0.645,0.648,0.651,0.65375,0.65675,0.6595,0.6625,0.6655,0.6685,0.6715,0.67425,0.67725,0.68025,0.68325,0.68625,0.68925,0.69225,0.69525,0.69825,0.7015,0.7045,0.7075,0.7105,0.71375,0.71675,0.71975,0.723,0.726,0.72925,0.73225,0.7355,0.7385,0.74175,0.745,0.748,0.75125,0.7545,0.75775,0.76075,0.764,0.76725,0.7705,0.77375,0.777,0.78025,0.7835,0.787,0.79025,0.7935,0.79675,0.8,0.8035,0.80675,0.81,0.8135,0.81675,0.82025,0.8235,0.827,0.8305,0.83375,0.83725,0.84075,0.844,0.8475,0.851,0.8545,0.858,0.8615,0.865,0.8685,0.872,0.8755,0.879,0.8825,0.88625,0.88975,0.89325,0.89675,0.9005,0.904,0.90775,0.91125,0.915,0.9185,0.92225,0.92575,0.9295,0.93325,0.937,0.9405,0.94425,0.948,0.95175,0.95525,0.959,0.96275,0.9665,0.97,0.97375,0.9775,0.98125,0.98475,0.9885,0.99225,0.996,0.9995]
lum6_before = [6.536e+39,1.204e+40,1.147e+40,9.059e+39,1.021e+40,1.032e+40,9.861e+39,1.089e+40,1.112e+40,1.147e+40,9.747e+39,1.032e+40,9.976e+39,9.861e+39,1.055e+40,9.976e+39,9.976e+39,1.055e+40,1.066e+40,1.101e+40,1.078e+40,1.204e+40,9.976e+39,1.170e+40,1.112e+40,1.089e+40,8.829e+39,1.170e+40,1.158e+40,1.135e+40,1.284e+40,1.184e40,1.101e+40,9.517e+39,1.101e+40,1.078e+40,1.112e+40,9.861e+39,1.124e+40,9.976e+39,1.043e+40,1.135e+40,1.273e+40,1.204e+40,1.043e+40,1.215e+40,1.066e+40,1.066e+40,9.976e+39,1.124e+40,1.158e+40,1.066e+40,1.009e+40,1.215e+40,1.284e+40,1.170e+40,1.283e+40,1.172e+40,3.736e+40,4.084e+40,4.894e+40,2.420e+40,3.530e+40,4.257e+40,5.487e+40,3.926e+40,8.162e+40,5.316e+40,4.851e+40,3.894e+40,5.142e+40,6.162e+40,6.380e+40,5.690e+40,1.726e+41,1.381e+41,2.321e+41,2.233e+41,1.288e+41,2.590e+41,2.138e+41,1.826e+41,1.997e+41,8.497e+40,6.721e+40,1.470e+41,3.149e+41,3.011e+41,1.586e+41,1.486e+41,2.280e+41,3.740e+41,2.841e+41,5.853e+41,5.729e+41,4.928e+41,5.831e+41,2.212e+41,2.348e+41,2.237e+41,1.833e+41,7.830e+40,4.964e+41,6.751e+41,2.170e+41,4.188e+41,4.880e+41,1.910e+41,4.549e+41,8.969e+41,1.035e+42,1.149e+42,6.534e+41,5.351e+41,4.183e+41,5.872e+41,1.063e+41,4.911e+41,4.956e+41,4.925e+41,4.813e+41,7.475e+41,6.230e+41,7.555e+41,3.891e+41,5.785e+41,1.671e+41,3.039e+41,3.483e+41,5.244e+41,4.071e+41,1.935e+41,4.053e+41,6.321e+41,8.068e+41,3.446e+41,5.061e+41,6.306e+41,3.452e+41,4.294e+41,4.689e+41,6.652e+41,1.097e+42,1.523e+42,1.981e+42,1.718e+42,1.769e+42,1.685e+42,1.726e+42,1.949e+42,2.326e+42,2.647e+42,3.019e+42,2.831e+42,2.703e+42,3.213e+42,3.910e+42,3.871e+42,4.237e+42,4.255e+42,3.895e+42]


red6 = np.loadtxt(pre + '6.txt') #np.loadtxt(pre + '6v2.txt')
# bad, replace this with something good.
red4 = np.loadtxt(pre + '4-fid.txt')
days4 = [0.505,0.515,0.525,0.5325,0.5425,0.5525,0.56,0.57,0.58,0.59,0.5975,0.6075,0.6175,0.625,0.635,0.645,0.6525,0.6625,0.6725,0.68,0.69,0.7,0.71,0.7175,0.7275,0.7375,0.745,0.755,0.765,0.7725,0.7825,0.7925,0.8,0.81,0.82,0.83,0.8375,0.8475,0.8575,0.865,0.875,0.885,0.8925,0.9025,0.9125,0.92,0.93,0.94,0.95,0.9575,0.9675,0.9775,0.985,0.995,1.005,1.0125,1.0225,1.0325,1.04,1.05,1.06,1.0675,1.0775,1.0875,1.0975,1.105,1.115,1.125,1.1325,1.1425,1.1525,1.16,1.17,1.18,1.1875,1.1975,1.2075,1.2175,1.225,1.235,1.245,1.2525,1.2625,1.2725,1.28,1.29,1.3,1.3075,1.3175,1.3275,1.335,1.345,1.355,1.365,1.375,1.3825,1.3925,1.4025,1.41,1.42,1.43,1.4375,1.4475,1.4575,1.4675,1.475,1.485,1.495,1.5025,1.5125,1.5225,1.53,1.54,1.55,1.5575,1.5675,1.5775,1.585,1.5975,1.6075,1.615,1.625,1.635,1.6425,1.6525,1.6625,1.6725,1.68,1.69,1.7,1.7075,1.7175,1.7275,1.735,1.745,1.755,1.7625,1.7725,1.7825,1.7925,1.8,1.81,1.82,1.8275,1.8375]
days4_CHR = [0.7825,0.79,0.8,0.81,0.8175,0.8275,0.8375,0.845,0.855,0.865,0.875,0.8825,0.8925,0.9025,0.91,0.92,0.93,0.9375,0.9475,0.9575,0.965,0.975,0.985,0.995,1.0025,1.0125,1.0225,1.03,1.04,1.05,1.0575,1.0675,1.0775,1.085,1.095,1.105,1.115,1.1225,1.1325,1.1425,1.15,1.16,1.17,1.1775,1.1875,1.1975,1.205,1.215,1.225,1.235,1.2425,1.2525,1.2625,1.27,1.28,1.29,1.2975,1.3075,1.3175,1.325,1.335,1.345]
red4_CHR = np.loadtxt(pre + '4-S60ComptonHires.txt')

#plt.plot(days6_before, lum6_before, c = 'tab:red')
#plt.plot(red6[0], red6[1], c = 'tab:red', label = '6')
plt.plot(days4, red4, c = 'g', label = '4 - fiducial')
plt.plot(days4_CHR, red4_CHR, c = 'tab:green', label = '4 - Compton HiRes')

Ledd4 = 1.26e38 * 1e4
Ledd6 = 1.26e38 * 1e6
plt.axhline(Ledd4, c = 'k', linestyle = 'dotted')
plt.axhline(Ledd6, c = 'k', linestyle = 'dotted')
plt.text(0.45, Ledd4 * 1.2, 'L$_{edd}$ $10^4 M_{\odot}$', fontsize = 16)
plt.text(0.45, Ledd6 * 1.2, 'L$_{edd}$ $10^6 M_{\odot}$', fontsize = 16)

plt.yscale('log')
plt.xlabel('Time [t/t$_{FB}$]')
plt.ylabel('Luminosity [erg/s]')
plt.title('Flux Limited Diffusion')
plt.grid()
plt.legend(loc = 'best')
plt.savefig('Final plot/lightcurves46.png')
plt.show()