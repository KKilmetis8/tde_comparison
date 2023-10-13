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
    # Minimum we need is 3.99e-22, Elad's lnrho stops at 1e-10
    rho_min = np.log(3.99e-22)
    rho_max = np.log(1e-10)
    expanding_rho = np.arange(rho_min,rho_max, 0.2)
    table_expansion = np.zeros( (len(lnT), len(expanding_rho) ))
    for i, T in enumerate(lnT):
        opacity_col = lnk_scatter[i]
        extra = CubicSpline(lnrho, opacity_col, bc_type='natural')
        for j, rho in enumerate(expanding_rho):
            table_expansion[i,j] = extra(rho)
    np.savetxt('scatter_expansion.txt', table_expansion)

