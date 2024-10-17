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