#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gives ray which are logspaced. Around photosphere they sould be 1.
Created on Tue Oct 10 10:19:34 2023
Also does midplane
@author: paola, konstantinos

"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

from src.Utilities.isalice import isalice
alice, plot = isalice()

import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from src.Utilities.selectors import select_prefix
#%% Constants & Converter
Msol_to_g = 1.989e33 # [g]
Rsol_to_cm = 6.957e10 # [cm]
den_converter = Msol_to_g / Rsol_to_cm**3

def grid_maker(fix, m, check, what, mass_weigh, x_num, y_num, z_num = 100):
    """ Outputs are in in solar units """
    fix = str(fix)
    Mbh = 10**m 
    Rt =  Mbh**(1/3) # Msol = 1, Rsol = 1
    apocenter = 2 * Rt * Mbh**(1/3)  # There is m_* hereeee
    
    # Load data
    pre = select_prefix(m, check)
    X = np.load(pre + fix + '/CMx_' + fix + '.npy')
    Y = np.load(pre + fix + '/CMy_' + fix + '.npy')
    Z = np.load(pre + fix + '/CMz_' + fix + '.npy')
    if mass_weigh:
        Mass = np.load(pre + fix + '/Mass_' + fix + '.npy')    
    
    if what == 'density':
        projected_quantity = np.load(pre + fix + '/Den_' + fix + '.npy')
        projected_quantity *= den_converter 
    elif what == 'temperature':
        projected_quantity = np.load(pre + fix + '/T_' + fix + '.npy')
    else:
        raise ValueError('Hate to break it to you champ \n \
                         but we don\'t have that quantity')

    # make a tree
    sim_value = [X, Y, Z] 
    sim_value = np.transpose(sim_value) #array of dim (number_points, 3)
    sim_tree = KDTree(sim_value) 
    
    # Ensure that the regular grid cells are smaller than simulation cells
    # NOTE: Shouldn't this conform to what the projector asks?
    x_start = -40 * Rt # -apocenter
    x_stop = 10 * Rt
    if m == 6:
        y_start = -20 * Rt
        y_stop = 30 * Rt
    if m == 5:
        y_start = -20 * Rt
        y_stop = 30 * Rt
    if m == 4:
        y_start = -30 * Rt # -0.5*apocenter
        y_stop = 30 * Rt # 0.5*apocenter
    z_start = -2 * Rt
    z_stop = 2 * Rt
    
    x_radii = np.linspace(x_start, x_stop, x_num) #simulator units
    y_radii = np.linspace(y_start, y_stop, y_num) #simulator units
    z_radii = np.linspace(z_start, z_stop, z_num) #simulator units

    gridded_indexes =  np.zeros(( len(x_radii), len(y_radii), len(z_radii) ))
    gridded_den =  np.zeros(( len(x_radii), len(y_radii), len(z_radii) ))
    gridded_mass =  np.zeros(( len(x_radii), len(y_radii), len(z_radii) ))
    for i in range(len(x_radii)):
        last_progress = 1
        for j in range(len(y_radii)):
            for k in range(len(z_radii)):
                queried_value = [x_radii[i], y_radii[j], z_radii[k]]
                _, idx = sim_tree.query(queried_value)
                                    
                # Store
                gridded_indexes[i, j, k] = idx
                gridded_den[i, j, k] = projected_quantity[idx]
                if mass_weigh:
                    gridded_mass[i,j, k] = Mass[idx]
        
        # Progress Check
        progress = int(100 * i/len(x_radii))
        if progress != last_progress:
            last_progress = progress    
            print('Progress: {:1.0%}'.format(i/len(x_radii)))
        
    return gridded_indexes, gridded_den, gridded_mass, x_radii, y_radii, z_radii

 
if __name__ == '__main__':
    m = 4
    Mbh = 10**m 
    Rt =  Mbh**(1/3) # Msol = 1, Rsol = 1
    check = 'fid'
    what = 'temperature'
    gridded_indexes, grid_den, grid_mass, xs, ys, zs = grid_maker(394, m, check, what,
                                                                  False, 200, 200, 100)
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
                
        den_plot = np.nan_to_num(grid_den, nan = -1, neginf = -1)
        den_plot = np.log10(den_plot[:,:, len(zs)//2])
        den_plot = np.nan_to_num(den_plot, nan = 0, neginf= 0)
        print(den_plot)
        # den_plot[den_plot < 0.1] = 0
        
        ax.set_xlabel(r' X/$R_T$ [R$_\odot$]', fontsize = 14)
        ax.set_ylabel(r' Y/$R_T$ [R$_\odot$]', fontsize = 14)
        img = ax.pcolormesh(xs/Rt, ys/Rt, den_plot.T, cmap = 'cet_fire',
                            vmin = vmin, vmax = vmax)
        cb = plt.colorbar(img)
        cb.set_label(cb_text, fontsize = 14)
        ax.set_title('Midplane', fontsize = 16)
        ax.set_xlim(-40,)
        ax.set_ylim(-30,)
        ax.plot(np.array(photo_x) / Rt, np.array(photo_y) / Rt, 
                marker = 'o', color = 'springgreen', linewidth = 3)