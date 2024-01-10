#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 16:26:39 2023

@author: konstantinos

NOTES FOR OTHERS
- T, rho are in CGS
"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

import numpy as np
from src.Opacity.opacity_table import opacity 

# All units are cgs (NO log)
loadpath = 'src/Opacity/'
Tcool = np.loadtxt(loadpath + 'Tcool_ext.txt')
sig_abs = np.loadtxt(loadpath + 'sigma_abs.txt')

c = 2.99792458e10 #[cm/s]
alpha = 7.5646 * 10**(-15) # radiation density [erg/cm^3K^4]

def old_opacity(T, rho, kind) -> float:
    '''
    Return the rosseland mean opacity in [cgs], given a value of density,
    temperature and a kind of opacity. If ln = True, then T and rho are
    lnT and lnrho. Otherwise we convert them.
    
     Parameters
     ----------
     T : float,
         Temperature in [cgs].
     rho : float,
         Density in [cgs].
     kind : str,
         The kind of opacities. Valid choices are:
         rosseland, plank or effective.
     
    Returns
    -------
    opacity : float,
        The rosseland mean opacity in [cgs].
    '''    
    T = float(T)
    rho = float(rho)
    n = rho * 0.9 / (1.67e-24)

    interp_sig_abs = np.interp(T, Tcool, sig_abs)
    #print('check interp', interp_sig_abs)
    k_a = interp_sig_abs * n**2
    
    k_s = 0.34 * rho

    # Pick Opacity & Use Interpolation Function
    if kind == 'planck':
        kapparho = k_a
    
    elif kind == 'scattering':
        kapparho = k_s

    elif kind == 'effective':
        # STEINBERG & STONE (9) (Rybicky & Lightman eq. 1.98)
        kapparho = np.sqrt(3 * k_a * (k_a + k_s)) 
    
    elif kind == 'red':
        kapparho = k_a + k_s
        
    return kapparho

if __name__ == '__main__':
    test_newtab = opacity(1e6, 1e-10, 'planck', False)
    test = old_opacity(1e6, 1e-10, 'planck')
    print('new table', test_newtab)
    print('old table', test)
