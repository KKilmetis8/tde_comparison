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
import healpy as hp


from src.Calculators.THREE_tree_caster import grid_maker
import src.Utilities.selectors as s
import src.Utilities.prelude as c

from src.Utilities.parser import parse
from src.Utilities.isalice import isalice
alice, plot = isalice()

def find_sph_coord(r, theta, phi):
    x = r * np.sin(np.pi-theta) * np.cos(phi) #Elad has just theta
    y = r * np.sin(np.pi-theta) * np.sin(phi)
    z = r * np.cos(np.pi-theta)
    return [x,y,z]

def equator_photo(rays_photo): 
    # rays_photo = rays_photo/c.Rsol_to_cm # to solar unit to plot
    
    # Make zs
    Zs = []
    for iobs in range(0,192):
        theta, phi = hp.pix2ang(4, iobs) # theta in [0,pi], phi in [0,2pi]
        r_ph = rays_photo[iobs]
        xyz_ph = find_sph_coord(r_ph, theta, phi)
        Zs.append(xyz_ph[2])
    
    # Sort and keep 16 closest to equator
    idx_z_sorted = np.argsort(np.abs(Zs))
    print(idx_z_sorted)
    size = 16
    plus_x = []
    neg_x = []
    plus_y = []
    neg_y = []
    photo_x = []
    photo_y = []
    
    for j, iz in enumerate(idx_z_sorted):
        if j == size:
            break
        theta, phi = hp.pix2ang(4, iz) # theta in [0,pi], phi in [0,2pi]
        r_ph = rays_photo[iz]
        xyz_ph = find_sph_coord(r_ph, theta, phi)
        if xyz_ph[1]>0:
            plus_x.append(xyz_ph[0])
            plus_y.append(xyz_ph[1])
        else:
            neg_x.append(xyz_ph[0])
            neg_y.append(xyz_ph[1])
    
    # Arrays so they can be indexed
    plus_x = np.array(plus_x)
    plus_y = np.array(plus_y)
    neg_x = np.array(neg_x)
    neg_y = np.array(neg_y)
    
    # Sort to untangle
    theta_plus = np.arctan2(plus_y,plus_x)
    cos_plus = np.cos(theta_plus)
    psort = np.argsort(cos_plus)
    
    theta_neg = np.arctan2(neg_y,neg_x)
    cos_neg = np.cos(theta_neg)
    nsort = np.argsort(cos_neg)

    # Combine correct
    photo_x = np.concatenate((plus_x[psort],  np.flip(neg_x[nsort],  )))
    photo_y = np.concatenate((plus_y[psort],  np.flip(neg_y[nsort], )))
    
    # Close the loop
    photo_x = np.append(photo_x, photo_x[0])
    photo_y = np.append(photo_y, photo_y[0])
    
    return photo_x, photo_y

#%%
alice, plot = isalice()
if alice:
    pre = '/home/kilmetisk/data1/TDE/'
else:
    pre = ''
# Constants & Converter

@numba.njit
def projector(gridded_den, gridded_mass, mass_weigh, x_radii, y_radii, z_radii, what):
    """ Project density on XY plane. NB: to plot you have to transpose the saved data"""
    # Make the 3D grid 
    flat_den =  np.zeros(( len(x_radii), len(y_radii) ))
    for i in range(len(x_radii)):
        for j in range(len(y_radii)):
            mass_zsum = 0
            step = 0
            for k in range(len(z_radii) - 1): # NOTE SKIPPING LAST Z PLANE
                dz = (z_radii[k+1] - z_radii[k]) # * Rsol_to_cm
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
        if args.single:
            fixes = [args.only]
        else:
            fixes = np.arange(args.first, args.last + 1)
        save = True
        picset = 'normal'
        m = int(np.log10(float(args.blackhole)))
        star = 'MONO AEK'
        check = 'OPOIOS SAS GAMAEI EINAI AEK'
        what = 'density'
    else:
        # Choose simulation
        m = 4
        picset = '5zoomout'
        opac_kind = 'LTE'
        check = 'fid'
        mstar = 0.5
        rstar = 0.47
        Rt = rstar * (10**m/mstar)**(1/3) 
        beta = 1
        what = 'density' # temperature or density
        save = False
        fixes = [348]
        args = None

    for fix in fixes:
        _, grid_den, grid_mass, xs, ys, zs = grid_maker(fix, m, 
                                                        200, 200, 10, 
                                                        picturesetting = picset,
                                                        parsed = args)
        flat_den = projector(grid_den, grid_mass, False,
                            xs, ys, zs, what)
        
        den_plot = np.nan_to_num(flat_den, nan = -5, neginf = -5)
        den_plot *= c.Msol_to_g / c.Rsol_to_cm**2
        den_plot = np.log10(den_plot)
        den_plot = np.nan_to_num(den_plot, neginf= 0)

        if save:
            if alice:
                pre = f'/home/kilmetisk/data1/TDE/tde_comparison/data/denproj/paper'
                np.savetxt(f'{pre}/{m}{picset}{fix}.txt', den_plot)
                np.savetxt(f'{pre}/{m}{picset}x.txt', xs)
                np.savetxt(f'{pre}/{m}{picset}y.txt', ys)
            else:
                np.savetxt(f'data/denproj{m}_{fix}.txt', flat_den) 
                np.savetxt(f'data/xarray{m}.txt', xs) 
                np.savetxt(f'data/yarray{m}.txt', ys) 

#%% Plot
        if plot:
            import colorcet
            fig, ax = plt.subplots(1,1)
            plt.rcParams['figure.figsize'] = [8, 4]
            
            # Clean
            den_plot = np.nan_to_num(flat_den, nan = -3, neginf = -3)
            den_plot *= c.Msol_to_g / c.Rsol_to_cm**2
            den_plot = np.log10(den_plot)
            den_plot = np.nan_to_num(den_plot, neginf= 0)
            
            # Specify
            if what == 'density':
                cb_text = r'Density [g/cm$^2$]'
                vmin = -5
                vmax = 2
            elif what == 'temperature':
                cb_text = r'Temperature [K]'
                vmin = 2
                vmax = 8
            else:
                raise ValueError('Hate to break it to you champ \n \
                                but we don\'t have that quantity')
                    

            ax.set_xlabel(r' X [AU]', fontsize = 14)
            ax.set_ylabel(r' Y [AU]', fontsize = 14)
            img = ax.pcolormesh(xs * c.Rsol_to_au, 
                                ys * c.Rsol_to_au, 
                                den_plot.T, 
                                cmap = 'cet_fire',
                                vmin = vmin, vmax = vmax)
            cb = plt.colorbar(img)
            cb.set_label(cb_text, fontsize = 14)
            

            # time = np.loadtxt(f'{m}/{fixes[0]}/tbytfb_{fixes[0]}.txt')
            # ax.set_title(f'10$^{m} M_\odot$ - {time:.2f} $t_\mathrm{{FB}}$',
            #              fontsize = 16)
            import pandas as pd
            df = pd.read_csv(f'data/photosphere/richex_photocolor{m}.csv', sep = ',',
                              comment = '#', header = None)
            
            # Find snap in photodata
            idx = np.argmin(np.abs(fix - df.iloc[:,0]))
            
            # snap time photo color obs_num
            # Photosphere data is a 3-tuple for each observer
            # The equatorial observers are 88:104
            def tuple_parse(strings):
                ''' parses "(1,2,3)" '''
                xs = np.zeros(len(strings))
                ys = np.zeros(len(strings))
                for i, string in enumerate(strings):
                    values = string.strip("()").split(", ")
                    tuple_values = tuple(np.float64(value.split("(")[-1].strip(")")) for value in values)
                    xs[i] = tuple_values[0]
                    ys[i] = tuple_values[1]
                return xs, ys
            good_obs = np.array([89, 90, 91, 92, 102, 103, 104, 89, 94]) + 4
            good_obs = np.arange(88, 104+1) + 4 + 192
            good_obs = np.arange(88, 104) + 4# + 192
            photo_x4, photo_y4 = tuple_parse(df.iloc[idx][good_obs])
            ax.plot(photo_x4 * c.Rsol_to_au, photo_y4 * c.Rsol_to_au, 
                    '-o', c = 'c', markersize = 3, label = 'Photosphere')
            ax.set_title(f'{m} Snap {fix}')
            # ax.set_ylim(-100,100)
            # ax.set_xlim(-200, 100)


