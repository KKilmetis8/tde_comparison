#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 16:26:39 2023

@author: konstantinos
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

# All units are ln[cgs]
loadpath = 'src/Optical_Depth/'
lnT = np.loadtxt(loadpath + 'T.txt')
lnrho = np.loadtxt(loadpath + 'rho.txt')
lnk_ross = np.loadtxt(loadpath + 'ross.txt')
lnk_plank = np.loadtxt(loadpath + 'plank.txt')
lnk_scatter = np.loadtxt(loadpath + 'scattering.txt')

lnk_scatter_inter = RegularGridInterpolator( (lnT, lnrho), lnk_scatter)
lnk_ross_inter = RegularGridInterpolator( (lnT, lnrho), lnk_ross)
lnk_plank_inter = RegularGridInterpolator( (lnT, lnrho), lnk_plank)

def opacity(rho, T, kind, ln = False):
    '''
    Parameters
    ----------
    rho : float,
        Density in [cgs].
    T : float,
        Temperature in [cgs].
    kind : str,
        The kind of opacities. Valid choices are:
        rosseland, plank or effective.
    log : bool,
        If True, then T and rho are lnT and lnrho

    Returns
    -------
    opacity : float,
        The rosseland mean opacity in [cgs].
    '''    
    if ln:
        
        # Pick Opacity & Use Interpolation Function
        if kind == 'rosseland':
            ln_opacity = lnk_ross_inter((T, rho))
            
        elif kind == 'plank':
            ln_opacity = lnk_plank_inter((T, rho))
            
        elif kind == 'effective':
            plank = lnk_plank_inter((T, rho))
            scattering = lnk_scatter_inter((T, rho))
            
            # Rybicky & Lightman eq. 1.98
            ln_opacity = np.sqrt(plank * (plank + scattering)) 
        else:
            print('Invalid opacity type. Try rosseland, plank or effective.')
            return 1
        
    else:
        # Turn to ln 
        T = np.log(T)
        rho = np.log(rho)
        
        # Pick Opacity & Use Interpolation Function
        if kind == 'rosseland':
            ln_opacity = lnk_ross_inter((T, rho))
            
        elif kind == 'plank':
            ln_opacity = lnk_plank_inter((T, rho))
            
        elif kind == 'effective':
            plank = lnk_plank_inter((T, rho))
            scattering = lnk_scatter_inter((T, rho))
            
            # Rybicky & Lightman eq. 1.98
            ln_opacity = np.sqrt(plank * (plank + scattering)) 
            
    # Remove the ln
    opacity = np.exp(ln_opacity)
    
    return opacity
    
