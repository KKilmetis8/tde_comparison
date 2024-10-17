#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 12:38:26 2024

@author: konstantinos
"""

import sympy as sym

# Symbols
e, ep, j, jp, M = sym.symbols('e e_p j j_p M')

# Circularization curve + Euclidian distance
j = M/sym.sqrt(2) * e**(-1/2)
d =  (e - ep)**2 + (j - jp)**2 # distance sq.

# Diff
# dprime = sym.diff(d, e)
# dprimeprime = sym.diff(dprime, e)

# Turn to functions
d_f = sym.lambdify( (e, ep, jp, M), d, 'numpy')
# dprime_f = sym.lambdify( (e, ep, jp, M), dprime, 'numpy')
# dprimeprime_f = sym.lambdify( (e, ep, jp, M), dprimeprime, 'numpy')

#%%
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**3 - 2*x**2 + 8*x - 10
import numba

@numba.njit
def d_prime(x, epsilon, j, Mbh):
    inv_sqrt2 = 1/np.sqrt(2)
    # oros0 = ((x-epsilon)**2 + (Mbh*inv_sqrt2*x**(-1/2) - j)**2)**(-1)/2
    oros1 = 2*(x-epsilon)
    par21 = Mbh*inv_sqrt2 * x**(-3/2)
    par22 = Mbh*inv_sqrt2 * x**(-1/2) - j
    return oros1 - par21*par22

@numba.njit
def d_primeprime(x, epsilon, j, Mbh):
    inv_sqrt2 = 1/np.sqrt(2)
    oros1 = 0.25 * Mbh**2 * x**(-3)
    par21 = Mbh * inv_sqrt2 * x**(-0.5) - j
    par22 = 0.75 * np.sqrt(2) * Mbh * x**(-2.5)
    return oros1 - par21*par22

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
        print('No roots here bucko')
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
        
def root_finder(a, b, f, fp, args, switch = 1e-2, tol=1e-4, small=1e-29):
    sol = regula_falsi(a, b, f, args, tolerance=switch)
    #sol = newt_raph(sol, f, fp, args, tolerance=tol, small=small)
    return sol
#%%
if __name__ == '__main__':
    Mbh = 1e5
    rstar = 0.47
    mstar = 0.5
    Rt = rstar * (Mbh/mstar)**(1/3)
    Rp = Rt
    jp = np.sqrt(2*Rp*Mbh)
    delta_e = mstar/rstar * ( (Mbh/mstar)**(1/3) - 1 )
    xs = np.linspace(0.1*delta_e, 10*delta_e, num = 2000)
    epoint = 0.5*delta_e
    jpoint = jp
    args = (epoint, jpoint, Mbh)
    ys = [ d_f(x, *args) for x in xs]
    
    fig, ax = plt.subplots(1, 1, figsize = (4,3), dpi = 300)
    ax.plot(xs/delta_e, ys, c='k')
    sol = root_finder(a = 0.1*delta_e, b = 30*delta_e, f = d_prime, fp = d_primeprime, args = args)
    # sol = newt_raph(epoint, d_prime, d_primeprime, args)
    ax.plot(sol/delta_e, d_f(sol, *args), 'o', c='r')
    ax2 = ax.twinx()
    
    ysp = [ abs(d_prime(x, *args)) for x in xs]
    ax2.plot(xs/delta_e, ysp, c = 'k', ls = '--')
    ax2.axhline(0)
    ax2.plot(sol/delta_e, abs(d_prime(sol, *args)), '^', c='darkorange')
    
    ax.set_yscale('log')
    ax2.set_yscale('log')
    
    baseplot(Mbh, ecirc, delta_e)
    plt.plot([sol/delta_e, epoint/delta_e], [circ_locus(sol, Mbh)/jp, jpoint/jp], ':h', c='k', lw=1,
              markersize = 3)
    plt.plot(epoint/delta_e, jpoint/jp, 'h', c='b', markersize = 4)
    plt.plot(sol/delta_e, circ_locus(sol, Mbh)/jp, 'h', c='r', markersize = 4)
