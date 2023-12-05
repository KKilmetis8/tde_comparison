#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project quantities.

@author: paola

"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

import numpy as np
import matplotlib.pyplot as plt
import numba
from src.Calculators.tree_caster import grid_maker

#%% Constants & Converter
G = 6.6743e-11 # SI
Msol = 1.98847e30 # kg
Rsol = 6.957e8 # m
t = np.sqrt(Rsol**3 / (Msol*G )) # Follows from G = 1
c = 3e8 * t/Rsol # simulator units. Need these for the PW potential
c_cgs = 3e10 # [cm/s]
Msol_to_g = 1.989e33 # [g]
Rsol_to_cm = 6.957e10 # [cm]
den_converter = Msol_to_g / Rsol_to_cm**2

@numba.njit
def projector(gridded_den, gridded_mass, x_radii, y_radii, z_radii):
    # Make the 3D grid
    flat_den =  np.zeros(( len(x_radii), len(y_radii) ))
    # flat_mass =  np.zeros(( len(x_radii), len(y_radii) ))
    for i in range(len(x_radii)):
        for j in range(len(y_radii)):
            mass_zsum = 0
            for k in range(len(z_radii) - 1): # NOTE SKIPPING LAST Z PLANE
                dz = (z_radii[k+1] - z_radii[k]) * Rsol_to_cm
                mass_zsum += gridded_mass[i,j,k]
                flat_den[i,j] += gridded_den[i,j,k] * dz #* gridded_mass[i,j,k]
            #flat_den[i,j] = np.divide(flat_den[i,j], mass_zsum)
    return flat_den
 
if __name__ == '__main__':
    m = 6
    check = 'fid'
    _, gridded_den, gridded_mass, x_radii, y_radii, z_radii = grid_maker(844, m, check,
                                                                         100, 100)
    flat_den = projector(gridded_den, gridded_mass, x_radii, y_radii, z_radii)

#%% Plot
    plot = True
    if plot:
        import colorcet
        fig, ax = plt.subplots(1,1)
        plt.rcParams['text.usetex'] = True
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['figure.figsize'] = [6, 4]
        plt.rcParams['axes.facecolor']= 	'whitesmoke'
        
        den_plot = np.nan_to_num(flat_den, nan = -1, neginf = -1)
        den_plot = np.log10(den_plot)
        den_plot = np.nan_to_num(den_plot, neginf= 0)
  
        # ax.set_xlim(-15_000, 2000)
        # ax.set_ylim(-4_000, 4000)
        ax.set_xlabel(r' X [$R_\odot$]', fontsize = 14)
        ax.set_ylabel(r' Y [R$_\odot$]', fontsize = 14)
        img = ax.pcolormesh(x_radii, y_radii, den_plot.T, cmap = 'jet',
                            vmin = 0, vmax = 5)
        cb = plt.colorbar(img)
        cb.set_label(r'Density [g/cm$^2$]', fontsize = 14)
        ax.set_title('XY Projection', fontsize = 16)
        plt.show()
    