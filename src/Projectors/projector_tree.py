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
        save = False
        fixes = [308]
        args = None

    for fix in fixes:
        print(fix)
        _, grid_den, grid_mass, xs, ys, zs = grid_maker(fix, m, star, check,
                                                        100, 100, 10, False,
                                                        args)
        flat_den = projector(grid_den, grid_mass, False,
                            xs, ys, zs, what)

        if save:
            if alice:
                pre = f'/home/kilmetisk/data1/TDE/tde_comparison/data/denproj/{sim}'
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
                vmax = 5
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
            img = ax.pcolormesh(xs, ys, den_plot.T, cmap = 'cet_fire',
                                vmin = vmin, vmax = vmax)
            cb = plt.colorbar(img)
            cb.set_label(cb_text, fontsize = 14)
            
            photo_x4, photo_y4 = equator_photo(photo_elad)
            ax.plot(np.array(photo_x4), np.array(photo_y4), 
                ':o', color = 'm', linewidth = 3)
            # ax.plot(np.array(photos_dir_X[88:104]), np.array(photos_dir_Y[88:104]), 
            #     ':o', color = 'm', linewidth = 3)
            ax.text(-1500, 0, '100', c = 'c', fontsize = 16)
            ax.set_title('XY Projection', fontsize = 16)
            # ax.set_xlim(-2000, 500)
            # ax.set_ylim(-300, 300)
            #plt.savefig(f'{snap}T.png')
            #plt.show()

