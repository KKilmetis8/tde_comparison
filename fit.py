""""
@author: paola 
"""

import numpy as np
from scipy.optimize import curve_fit

m = 6
sigma = 5.6704e-5 #g/s^3K^4

def luminosity_BB(T, R):
    L = 4 * np.pi * sigma * T**4 * R**2
    return L

L_tilde = np.loadtxt('L_m' + str(m) + '.txt')


