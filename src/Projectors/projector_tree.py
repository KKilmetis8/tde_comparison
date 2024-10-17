#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project quantities.

@author: paola + konstantinos

"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')


import numpy as np
import matplotlib.pyplot as plt
import numba

from src.Calculators.THREE_tree_caster import grid_maker
import src.Utilities.selectors as s
from src.Utilities.parser import parse
from src.Utilities.isalice import isalice
alice, plot = isalice()

#%%
alice, plot = isalice()
if alice:
    pre = '/home/s3745597/data1/TDE/'
else:
    pre = ''
# Constants & Converter
Rsol_to_cm = 6.957e10 # [cm]

@numba.njit
def projector(gridded_den, gridded_mass, mass_weigh, x_radii, y_radii, z_radii, what):
    """ Project density on XY plane. NB: to plot you have to transpose the saved data"""
    # Make the 3D grid 
    flat_den =  np.zeros(( len(x_radii), len(y_radii) ))
    # flat_mass =  np.zeros(( len(x_radii), len(y_radii) ))
    for i in range(len(x_radii)):
        for j in range(len(y_radii)):
            mass_zsum = 0
            step = 0
            for k in range(len(z_radii) - 1): # NOTE SKIPPING LAST Z PLANE
                dz = (z_radii[k+1] - z_radii[k]) * Rsol_to_cm
                if mass_weigh:
                    mass_zsum += gridded_mass[i,j,k]
                    flat_den[i,j] += gridded_den[i,j,k] * dz * gridded_mass[i,j,k]
                else:
                    if what == 'density':
                        flat_den[i,j] += gridded_den[i,j,k] * dz
                    else: 
                        flat_den[i,j] += gridded_den[i,j,k]
                        step += 1
            if mass_weigh:
                flat_den[i,j] = np.divide(flat_den[i,j], mass_zsum)
            if what != 'density': 
                flat_den[i,j] /= step
    return flat_den
 
if __name__ == '__main__':

    if alice:
        args = parse()
        fixes = np.arange(args.first, args.last + 1)
        sim = args.name
        save = True
        m = 'AEK'
        star = 'MONO AEK'
        check = 'OPOIOS SAS GAMAEI EINAI AEK'
        what = 'density'
    else:
        # Choose simulation
        m = 5
        opac_kind = 'LTE'
        check = 'fid'
        mstar = 0.5
        if mstar == 0.5:
            star = 'half'
        else:
            star = ''
        rstar = 0.47
        beta = 1
        what = 'density' # temperature or density
        save = True
        fixes, days = s.select_snap(m, mstar, rstar, check, time = True)
        args = None

    for fix in fixes:
        print(fix)
        _, grid_den, grid_mass, xs, ys, zs = grid_maker(fix, m, star, check,
                                                        500, 500, 100, False,
                                                        args)
        flat_den = projector(grid_den, grid_mass, False,
                            xs, ys, zs, what)

        if save:
            if alice:
                pre = f'/home/s3745597/data1/TDE/tde_comparison/data/denproj/{sim}'
                np.savetxt(f'{pre}/denproj{sim}{fix}.txt', flat_den)
                np.savetxt(f'{pre}/xarray{sim}.txt', xs)
                np.savetxt(f'{pre}/yarray{sim}.txt', ys)
            else:
                np.savetxt(f'data/denproj{m}_{fix}.txt', flat_den) 
                np.savetxt(f'data/xarray{m}.txt', xs) 
                np.savetxt(f'data/yarray{m}.txt', ys) 

#%% Plot
        if plot:
            import colorcet
            fig, ax = plt.subplots(1,1)
            plt.rcParams['text.usetex'] = True
            plt.rcParams['figure.dpi'] = 300
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['figure.figsize'] = [6, 4]
            plt.rcParams['axes.facecolor']=     'whitesmoke'
            
            # Clean
            den_plot = np.nan_to_num(flat_den, nan = -1, neginf = -1)
            den_plot = np.log10(den_plot)
            den_plot = np.nan_to_num(den_plot, neginf= 0)
            
            # Specify
            if what == 'density':
                cb_text = r'Density [g/cm$^2$]'
                vmin = 0
                vmax = 6
            elif what == 'temperature':
                cb_text = r'Temperature [K]'
                vmin = 2
                vmax = 8
            else:
                raise ValueError('Hate to break it to you champ \n \
                                but we don\'t have that quantity')
                    
            # ax.set_xlim(-1, 10/20_000)
            # ax.set_ylim(-0.2, 0.2)
            ax.set_xlabel(r' X [$R_\odot$]', fontsize = 14)
            ax.set_ylabel(r' Y [$R_\odot$]', fontsize = 14)
            img = ax.pcolormesh(xs, ys, den_plot.T, cmap = 'cet_fire')
                                #vmin = vmin, vmax = vmax)
            cb = plt.colorbar(img)
            cb.set_label(cb_text, fontsize = 14)

            ax.set_title('XY Projection', fontsize = 16)
            plt.savefig(f'{snap}T.png')
            plt.show()
