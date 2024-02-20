#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project quantities.

@author: paola

"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

from src.Utilities.isalice import isalice
alice, plot = isalice()

import numpy as np
import matplotlib.pyplot as plt
import numba
import src.Utilities.selectors as s
from src.Calculators.THREE_tree_caster import grid_maker

# Constants & Converter
Rsol_to_cm = 6.957e10 # [cm]

@numba.njit
def projector(gridded_den, gridded_mass, x_radii, y_radii, z_radii):
    """ Project density on XY plane. NB: to plot you have to transpose the saved data"""
    # Make the 3D grid 
    flat_den =  np.zeros(( len(x_radii), len(y_radii) ))
    # flat_mass =  np.zeros(( len(x_radii), len(y_radii) ))
    for i in range(len(x_radii)):
        for j in range(len(y_radii)):
            mass_zsum = 0
            for k in range(len(z_radii) - 1): # NOTE SKIPPING LAST Z PLANE
                dz = (z_radii[k+1] - z_radii[k]) * Rsol_to_cm
                # mass_zsum += gridded_mass[i,j,k]
                flat_den[i,j] += gridded_den[i,j,k] * dz #* gridded_mass[i,j,k]
            #flat_den[i,j] = np.divide(flat_den[i,j], mass_zsum)
    return flat_den
 
if __name__ == '__main__':
    m = 4
    save = True 
    check = 'fid' 
    snapshots, days = s.select_snap(m, check)

    for snap in snapshots:
        _, gridded_den, gridded_mass, x_radii, y_radii, z_radii = grid_maker(snap, m, check,
                                                                         800, 800)
        flat_den = projector(gridded_den, gridded_mass, x_radii, y_radii, z_radii)

        if save:
            if alice:
                pre = '/home/s3745597/data1/TDE/'
                sim = f'{m}-{check}'
                np.savetxt(f'{pre}tde_comparison/data/denproj/denproj{sim}{snap}.txt', flat_den)
                np.savetxt(f'{pre}tde_comparison/data/denproj/xarray{sim}.txt', x_radii)
                np.savetxt(f'{pre}tde_comparison/data/denproj/yarray{sim}.txt', y_radii)
            else:
                np.savetxt(f'data/localdenproj{m}_{snap}.txt', flat_den) 
                np.savetxt(f'data/localxarray{m}.txt', x_radii) 
                np.savetxt(f'data/localyarray{m}.txt', y_radii) 

#%% Plot
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
                                vmin = 0, vmax = 7)
            cb = plt.colorbar(img)
            cb.set_label(r'Density [g/cm$^2$]', fontsize = 14)
            ax.set_title('XY Projection', fontsize = 16)
            plt.show()
    