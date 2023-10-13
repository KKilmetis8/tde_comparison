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
from scipy.interpolate import CubicSpline

# All units are ln[cgs]
loadpath = 'src/Optical_Depth/'
lnT = np.loadtxt(loadpath + 'T.txt')
lnrho = np.loadtxt(loadpath + 'rho.txt')
lnk_ross = np.loadtxt(loadpath + 'ross.txt')
lnk_planck = np.loadtxt(loadpath + 'planck.txt')
lnk_scatter = np.loadtxt(loadpath + 'scatter.txt')

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
    if rho > -23:
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
    else:
        # NOTE: Fair chance that this is VERY slow
        # Maybe do all 128x3 cubic splines at the start?
        # We could just straight up replace RegularGridInterpolator
        if kind == 'rosseland':
            # Get the column
            idx = np.argmin( np.abs( T - lnT ))
            opacity_col = lnk_ross[idx]
            
            # Inter/Extra polate
            extra = CubicSpline(lnrho, opacity_col, bc_type='natural')
            ln_opacity = extra(rho)
        
        elif kind == 'planck':
            # Get the column
            idx = np.argmin( np.abs( T - lnT ))
            opacity_col = lnk_planck[idx]
            
            # Inter/Extra polate
            extra = CubicSpline(lnrho, opacity_col, bc_type='natural')
            ln_opacity = extra(rho)       
            
        elif kind == 'effective':
            # Get the column
            idx = np.argmin( np.abs( T - lnT ))
            
            # sqrt(3 plank (plank + scattering) )s
            opacity_col = np.add(lnk_planck[idx], lnk_scatter[idx])
            opacity_col = np.multiply(3 * opacity_col, lnk_planck[idx])
            opacity_col = np.sqrt(opacity_col)
            # Inter/Extra polate
            extra = CubicSpline(lnrho, opacity_col, bc_type='natural')
            ln_opacity = extra(rho)
            
        elif kind == 'red':
            idx = np.argmin( np.abs( T - lnT ))
            opacity_col = np.add(lnk_planck[idx], lnk_scatter[idx])
            
            # Inter/Extra polate
            extra = CubicSpline(lnrho, opacity_col, bc_type='natural')
            ln_opacity = extra(rho)
            
    # Remove the ln
    opacity = np.exp(ln_opacity)
    
    return opacity

if __name__ == '__main__':
    opa = opacity(1e9, 1e-10, 'planck', ln = False)
    print(opa)
    
