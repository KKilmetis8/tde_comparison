#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: paola

Produce a new table already expanded, in order to interpolate here.
MISSING PART: create the new txt file for rho and opacities.
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

def extrapolation_table(rho, kind):
    extra = np.zeros(len(lnT))
    for i in range(len(lnT)):
        if kind == 'rosseland':
            opacity_row = lnk_ross[i]
        elif kind == 'planck':
            opacity_row = lnk_planck[i]     
        elif kind == 'effective':
            opacity_row = np.add(lnk_planck[i], lnk_scatter[i])
            opacity_row = np.multiply(3 * opacity_row, lnk_planck[i])
            opacity_row = np.sqrt(opacity_row)   
        elif kind == 'red':
            opacity_row = np.add(lnk_planck[i], lnk_scatter[i])
        cubicspl = CubicSpline(lnrho, opacity_row, bc_type='natural')
        extra[i] = cubicspl(rho)
    print(np.shape(extra))
    return extra

# def opacity_extr(rho_array, kind):
#     for i in range(len(lnT)):

#     extra = extrapolation_table(kind)
#     print(new_opacity)
#     return new_opacity

if __name__ == '__main__':
    expanding_rho = np.arange(-30,-22, 0.2)
    for i,rho in enumerate(expanding_rho):
        a = extrapolation_table(rho, 'rosseland')
