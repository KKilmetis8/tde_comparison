""""
@author: paola 
"""

import numpy as np
import matplotlib.pyplot as plt

sigma = 5.6704e-5 #g/s^3K^4

def luminosity_BB(T, R):
    L = 4 * np.pi * sigma * T**4 * R**2
    return L

data = np.loadtxt('Ltilda_m6.txt')
logspace = data[0]
L_tilde_n = data[1]

#### wrong idea... we don't have dependacny on n in luminosity_BB