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
from src.Utilities.selectors import select_snap
from src.Calculators.THREE_tree_caster import grid_maker

# Constants & Converter
Rsol_to_cm = 6.957e10 # [cm]

@numba.njit
def projector(gridded_den, gridded_mass, mass_weigh, x_radii, y_radii, z_radii):
    """ Project density on XY plane. NB: to plot you have to transpose the saved data"""
    # Make the 3D grid 
    flat_den =  np.zeros(( len(x_radii), len(y_radii) ))
    # flat_mass =  np.zeros(( len(x_radii), len(y_radii) ))
    for i in range(len(x_radii)):
        for j in range(len(y_radii)):
            mass_zsum = 0
            for k in range(len(z_radii) - 1): # NOTE SKIPPING LAST Z PLANE
                dz = (z_radii[k+1] - z_radii[k]) * Rsol_to_cm
                if mass_weigh:
                    mass_zsum += gridded_mass[i,j,k]
                    flat_den[i,j] += gridded_den[i,j,k] * dz * gridded_mass[i,j,k]
                else:
                    flat_den[i,j] += gridded_den[i,j,k] # * dz
            if mass_weigh:
                flat_den[i,j] = np.divide(flat_den[i,j], mass_zsum)
    return flat_den
 
if __name__ == '__main__':
    m = 4
    save = False 
    check = 'S60ComptonHires' 
    what = 'temperature' # temperature or density
    snapshots, days = select_snap(m, check)

    for snap in snapshots:
        _, grid_den, grid_mass, xs, ys, zs = grid_maker(snap, m, check, what, False,
                                                        100, 100, 10)
        flat_den = projector(grid_den, grid_mass, False,
                             xs, ys, zs)

        if save:
            if alice:
                pre = '/home/s3745597/data1/TDE/'
                sim = f'{m}-{check}'
                np.savetxt(f'{pre}tde_comparison/data/denproj/denproj{sim}{snap}.txt', flat_den)
                np.savetxt(f'{pre}tde_comparison/data/denproj/xarray{sim}.txt', xs)
                np.savetxt(f'{pre}tde_comparison/data/denproj/yarray{sim}.txt', ys)
            else:
                np.savetxt(f'data/localdenproj{m})_{snap}.txt', flat_den) 
                np.savetxt(f'data/localxarray{m}.txt', xs) 
                np.savetxt(f'data/localyarray{m}.txt', ys) 

#%% Plot
        if plot:
            import colorcet
            fig, ax = plt.subplots(1,1)
            plt.rcParams['text.usetex'] = True
            plt.rcParams['figure.dpi'] = 300
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['figure.figsize'] = [6, 4]
            plt.rcParams['axes.facecolor']= 	'whitesmoke'
            
            
            # Clean
            den_plot = np.nan_to_num(flat_den, nan = -1, neginf = -1)
            den_plot = np.log10(den_plot)
            den_plot = np.nan_to_num(den_plot, neginf= 0)
            
            # Specify
            if what == 'density':
                cb_text = r'Density [g/cm$^2$]'
                vmin = 0
                vmax = 7
            elif what == 'temperature':
                cb_text = r'Temperature [K]'
                vmin = 0
                vmax = 7
            else:
                raise ValueError('Hate to break it to you champ \n \
                                 but we don\'t have that quantity')
                    
            # ax.set_xlim(-15_000, 2000)
            # ax.set_ylim(-4_000, 4000)
            ax.set_xlabel(r' X [$R_\odot$]', fontsize = 14)
            ax.set_ylabel(r' Y [R$_\odot$]', fontsize = 14)
            img = ax.pcolormesh(xs, ys, den_plot.T, cmap = 'cet_fire',
                                vmin = vmin, vmax = vmax)
            cb = plt.colorbar(img)
            cb.set_label(cb_text, fontsize = 14)
   
            ax.set_title('XY Projection', fontsize = 16)
            plt.show()
    