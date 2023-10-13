#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 16:26:39 2023

@author: konstantinos

NOTES FOR OTHERS
- T, rho are in CGS
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
#from scipy.interpolate import CubicSpline

# All units are ln[cgs]
loadpath = 'src/Optical_Depth/'
lnT = np.loadtxt(loadpath + 'T.txt')
lnrho = np.loadtxt(loadpath + 'rho.txt')
# lnk_ross = np.loadtxt(loadpath + 'ross.txt')
# lnk_planck = np.loadtxt(loadpath + 'planck.txt')
# lnk_scatter = np.loadtxt(loadpath + 'scatter.txt')
lnk_ross = np.loadtxt(loadpath + 'ross_expansion.txt')
lnk_planck = np.loadtxt(loadpath + 'planck_expansion.txt')
lnk_scatter = np.loadtxt(loadpath + 'scatter_expansion.txt')

lnk_scatter_inter = RegularGridInterpolator( (lnT, lnrho), lnk_scatter)
lnk_ross_inter = RegularGridInterpolator( (lnT, lnrho), lnk_ross)
lnk_planck_inter = RegularGridInterpolator( (lnT, lnrho), lnk_planck)


def opacity(T, rho, kind, ln = True) -> float:
    '''
    Return the rosseland mean opacity in [cgs], given a value of density,
    temperature and and a kind of opacity. If ln = True, then T and rho are
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
     log : bool,
         If True, then T and rho are lnT and lnrho, Default is True
     
    Returns
    -------
    opacity : float,
        The rosseland mean opacity in [cgs].
    '''    
    if not ln: 
        T = np.log(T)
        rho = np.log(rho)
        # Remove fuckery
        T = np.nan_to_num(T, nan = 0, posinf = 0, neginf= 0)
        rho = np.nan_to_num(rho, nan = 0, posinf = 0, neginf= 0)

    # Pick Opacity & Use Interpolation Function
    if kind == 'rosseland':
        ln_opacity = lnk_ross_inter((T, rho))
        
    elif kind == 'planck':
        ln_opacity = lnk_planck_inter((T, rho))
        
    elif kind == 'effective':
        planck = lnk_planck_inter((T, rho))
        scattering = lnk_scatter_inter((T, rho))
        
        # Rybicky & Lightman eq. 1.98 NO USE STEINGERG & STONE (9)
        ln_opacity = np.sqrt(3 * planck * (planck + scattering)) 
        
    elif kind == 'red':
        planck = lnk_planck_inter((T, rho))
        scattering = lnk_scatter_inter((T, rho))
        
        # Rybicky & Lightman eq. 1.98 NO USE STEINGERG & STONE (9)
        ln_opacity = planck + scattering
    else:
        print('Invalid opacity type. Try: rosseland / planck / effective.')
        return 1
    
    opacity = np.exp(ln_opacity)
    
    return opacity

if __name__ == '__main__':
    opa = opacity(1e9, 1e-10, 'planck', ln = False)
    print(opa)
    
