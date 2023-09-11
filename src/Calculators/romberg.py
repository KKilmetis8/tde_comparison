#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 20:28:52 2023

@author: konstantinos
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 20:01:40 2023

@author: konstantinos
"""

import numpy as np
import matplotlib.pyplot as plt
def f_x2(x):
    return x**2

def int_trapz(x, f, *args):
    h = (x[-1] - x[0]) / (len(x) - 1)
    integral = 0.5*(f(x[0], *args) + f(x[-1], *args))
    for i in range(1,len(x)-1):
        integral += f(x[i], *args)
    return h*integral

def midpoint(x,f, *args):
    # Special case
    if len(x) <= 2:
        h = (x[-1] - x[0]) / (len(x) - 1)
        mid = (x[1] - x[0])*0.5
        integral = f(mid, *args)
        return integral
    
    mid = (x[1] - x[0])*0.5
    # Make new interval with the midpoints
    nx = np.linspace(x[0]+mid, x[-1]+mid, num=len(x)) # nx is new x

    h = (nx[-1] - nx[0]) / ( len(nx) - 1) 
    integral = 0
    for i in range(0, len(nx) - 1):
        integral += f(nx[i], *args)
    return integral*h

def romberg(a,b,f, *args, order=10):
    guesses = np.zeros((order, order))

    # Initial round of guesses through trapz.
    guesses[0,0] = int_trapz((a,b), f, *args)
    for i in range(1, order):
        interval = np.linspace(a,b, num=2**i+1)
        guesses[i,0] = int_trapz(interval, f, *args)
    
    # Combine
    for i in range(1, order):
        for j in range(1, i+1):
            par = 4**j 
            invpar = 1/(par - 1)
            guesses[i][j] = (par*guesses[i][j-1] - guesses[i-1][j-1])*invpar
            
    return guesses[-1,-1]

def romberg_mid(a,b,f,*args, order=10):
    guesses = np.zeros((order, order))

    # Initial round of guesses through mid.
    guesses[0,0] = midpoint((a,b), f, *args)
    for i in range(1, order):
        interval = np.linspace(a,b, num=2**i+1)
        guesses[i,0] = midpoint(interval, f, *args)
    
    # Combine
    for i in range(1, order):
        for j in range(1, i+1):
            par = 4**j 
            invpar = 1/(par - 1)
            guesses[i][j] = (par*guesses[i][j-1] - guesses[i-1][j-1])*invpar
            
    return guesses[-1,-1]
if __name__ == '__main__':
    a = np.linspace(0,1, num=5)
    plt.plot(a,f_x2(a), 'o', c='navy')
    b = midpoint(a, f_x2)
    plt.plot(b,f_x2(b), 'o', c='darkorange')
    
    ran1 = np.linspace(1, 5, num=2**5 +1)
    ys = f_x2(ran1)
    real = 5**3/3 -1/3
    print('Real:', real)
    
    rom_res = romberg(1,5, f_x2)
    rom_mid_res = romberg_mid(1,5, f_x2)
    print('Trapz Error %.2e' % np.abs(real - int_trapz(ran1, f_x2)))
    print('np Error: %.2e' % np.abs(real - np.trapz(ys, x=ran1)))
    # print('Romberg Error: %.2e' % np.abs(real - romberg_trapz(1,5, f_x2)))
    print('Romberg Error: %.2e' % np.abs(real - rom_res))
    print('Midpoint Romberg: %.2e' % np.abs(real - rom_mid_res))