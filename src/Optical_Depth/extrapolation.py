#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: paola

Produce a new table already expanded, in order to interpolate here.
MISSING PART: create the new txt file for rho and opacities.
"""

import numpy as np
from scipy.interpolate import CubicSpline

# All units are ln[cgs]
loadpath = 'src/Optical_Depth/'
lnT = np.loadtxt(loadpath + 'T.txt')
lnrho = np.loadtxt(loadpath + 'rho.txt')
lnk_ross = np.loadtxt(loadpath + 'ross.txt')
lnk_planck = np.loadtxt(loadpath + 'planck.txt')
lnk_scatter = np.loadtxt(loadpath + 'scatter.txt')


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
    rho_max = np.log(8e-11)
    expanding_rho = np.arange(rho_min,rho_max, 0.2)
    expanding_rho = np.arange(1,5)
    colum_expanded_rho = len(expanding_rho) + len(lnrho)
    table_expansion = np.zeros( (len(lnT), colum_expanded_rho ))
    for i, T in enumerate(lnT):
        opacity_col = lnk_ross[i] # line to change
        extra = CubicSpline(lnrho, opacity_col, bc_type='natural')
        for j, rho in enumerate(expanding_rho):           
            opi = extra(rho)
            if opi > 0 :
                table_expansion[i,j] = opi
        for j in range(len(expanding_rho),colum_expanded_rho):
            table_expansion[i,j] = opacity_col[j-len(expanding_rho)]
    
      
    # print(np.shape(table_expansion))
    # np.savetxt(loadpath + 'ross_expansion.txt', table_expansion)

    #all_rhos = np.concatenate((expanding_rho, lnrho))
    #np.savetxt(loadpath + 'rho_expansion.txt', all_rhos)
   