#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 17:26:20 2024

@author: konstantinos
"""
import numpy as np
import numba

@numba.njit
def newt_raph(x, f, fp, args,
              tolerance = 1e-10,
              small = 1e-29,
              max_iter = int(1e6)):
    x_new = x 
    nr_step = 0
    for i in range(max_iter):
        # Func Evals
        y = f(x, *args)
        yp = fp(x, *args)
        
        # Overflow Check
        if np.abs(yp) < small:
            print('Overflow')
            print('NR: ', nr_step)
            return x_new
        nr_step += 1
        
        # The actual bit
        x_new = x - y / yp

        # Convergance Check
        if np.abs(x_new-x) < tolerance:
            # print('NR: ', nr_step)
            return x_new
        
        # Again!
        x = x_new
        
@numba.njit
def regula_falsi(a, b, f, args,
       tolerance = 1e-2,
       max_iter = int(1e5)):
    rf_step = 0
    
    # Minimize f calls    
    fa = f(a, *args)
    fb = f(b, *args)
    
    # Sanity
    if fa * fb > 0:
        # print('No roots here bucko')
        return None
    
    # Initial guess
    ari = (a - b) * fa 
    par = fa - fb
    cc = a - ari/par
    fc = f(cc, *args)
    
    # Ensure convergenve (f(c) < tol)
    while np.abs(fc) > tolerance and rf_step<max_iter: 
        rf_step += 1

        ari = (a - b) * fa 
        par = fa - fb
        cc = a - ari/par
        fc = f(cc, *args)
        #print('f(c): %.1e' % f(c))
        if fa * fc < 0:
            b = cc
        else:
            a = cc
    #print('RF: ', rf_step) 
    return cc