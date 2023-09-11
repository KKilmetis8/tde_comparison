#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 16:26:39 2023

@author: konstantinos
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

# All units are ln[cgs]
loadpath = 'src/Ionization/'
lnT = np.loadtxt(loadpath + 'T.txt')
lnrho = np.loadtxt(loadpath + 'rho.txt')
lnk = np.loadtxt(loadpath + 'ross.txt') # rosseland mean

lnk_inter = RegularGridInterpolator( (lnT, lnrho), lnk)

def opacity(rho, T):
    
    # Turn to ln 
    # ln_temp = np.log(T)
    # ln_rho = np.log(rho)
    
    # Use interpolated function
    ln_opacity = lnk_inter((T, rho))
    
    # Remove the ln
    opacity = np.exp(ln_opacity)
    
    return opacity
    