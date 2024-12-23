#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make a 3D grid, searching for simulation data in the vicinity of the one chosen 
and storing X,Y,Z,Den.
Created on Tue Oct 10 10:19:34 2023
@authors: paola, konstantinos
"""
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

from src.Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    realpre = '/home/kilmetisk/data1/TDE/'
else:
    import sys
    sys.path.append('/Users/paolamartire/tde_comparison')
    realpre = ''

import numba
from tqdm import tqdm
# @numba.njit
# def masker(Den, X, Y, Z):
#     denmask = np.where((Den > 1e-12))[0]
#     X = X[denmask]
#     Y = Y[denmask]
#     Z = Z[denmask]
#     Den = Den[denmask]
#     return Den, X, Y, Z
import numexpr as ne
#%% Constants & Converter
Msol_to_g = 1.989e33 # [g]
Rsol_to_cm = 6.957e10 # [cm]
den_converter = Msol_to_g / Rsol_to_cm**3

def grid_maker(fix, m, star, check, x_num, y_num, z_num = 100, mass_weight=False,
               parsed = None):
    """ ALL outputs are in in solar units """

    if type(parsed) == type(None):
        Mbh = 10**m
        if 'star' == 'half':
            mstar = 0.5
            rstar = 0.47
        else:
            mstar = 1
            rstar = 1
        Rt = rstar * (Mbh/mstar)**(1/3) 
        apocenter = Rt * (Mbh/mstar)**(1/3)
        if alice:
            pre = f'{m}{star}-{check}/snap_{fix}'
        else: 
            pre = f'{m}/{fix}'
            # CM Position Data
        X = np.load(f'{pre}/CMx_{fix}.npy')
        Y = np.load(f'{pre}/CMy_{fix}.npy')
        Z = np.load(f'{pre}/CMz_{fix}.npy')
        Den = np.load(f'{pre}/Den_{fix}.npy')

    else:
        sim = parsed.name
        mstar = parsed.mass
        rstar = parsed.radius
        Mbh = parsed.blackhole
        Rt = rstar * (Mbh/mstar)**(1/3) 
        apocenter = Rt * (Mbh/mstar)**(1/3)
        X = np.load(f'{realpre}{sim}/snap_{fix}/CMx_{fix}.npy')
        Y = np.load(f'{realpre}{sim}/snap_{fix}/CMy_{fix}.npy')
        Z = np.load(f'{realpre}{sim}/snap_{fix}/CMz_{fix}.npy')
        Den = np.load(f'{realpre}{sim}/snap_{fix}/Den_{fix}.npy')

    WOW = 1
    x_start = -apocenter * WOW
    x_stop = 0.2 * apocenter * WOW
    # x_num = pixel_num # np.abs(x_start - x_stop)
    xs = np.linspace(x_start, x_stop, num = x_num )
    y_start = -0.2 * apocenter  * WOW
    y_stop = 0.2 * apocenter * WOW
    # y_num = pixel_num # np.abs(y_start - y_stop)
    ys = np.linspace(y_start, y_stop, num = y_num)
    z_start = -2 * Rt
    z_stop = 2 * Rt
    zs = np.linspace(z_start, z_stop, z_num) #simulator units


    # Density cut
    # denmask = np.where((Den > 1e-12))[0]
    denmask = ne.evaluate("Den > 1e-12")
    X = X[denmask]
    Y = Y[denmask]
    Z = Z[denmask]
    Den = Den[denmask]
    # Den, X, Y, Z = masker(Den, X, Y, Z)
    
    sim_value = [X, Y, Z] 
    sim_value = np.transpose(sim_value) #array of dim (number_points, 3)
    sim_tree = KDTree(sim_value) 

    gridded_indexes =  np.zeros(( len(xs), len(ys), len(zs) ))
    gridded_den =  np.zeros(( len(xs), len(ys), len(zs) ))
    gridded_mass =  np.zeros(( len(xs), len(ys), len(zs) ))
    for i in tqdm(range(len(xs))):
        for j in range(len(ys)):
            for k in range(len(zs)):
                queried_value = [xs[i], ys[j], zs[k]]
                _, idx = sim_tree.query(queried_value)
                                    
                # Store
                gridded_indexes[i, j, k] = idx
                gridded_den[i, j, k] = Den[idx]
    #             gridded_mass[i,j, k] = Mass[idx]
    # den_cast = np.divide(den_cast, gridded_mass)
    # den_cast = np.sum(gridded_den, axis=2) / 100

    # # Remove bullshit and fix things
    # den_cast = np.nan_to_num(den_cast.T)
    # den_cast = np.log10(den_cast) # we want a log plot
    # den_cast = np.nan_to_num(den_cast, neginf=0) # fix the fuckery
        
    # # Color re-normalization
    # den_cast[den_cast<0.2] = 0
    # den_cast[den_cast>5] = 5

    # return xs/apocenter, ys/apocenter, den_cast, apocenter# , days
    # 
    return gridded_indexes, gridded_den, gridded_mass, xs, ys, zs

 
if __name__ == '__main__':
    m = 4
    Mbh = 10**m 
    Rt =  Mbh**(1/3) # Msol = 1, Rsol = 1
    check = 'fid'
    what = 'density'
    gridded_indexes, grid_den, grid_mass, xs, ys, zs = grid_maker(297, m,
                                                                  'half', check, 
                                                                  50, 50, 
                                                                  z_num = 10, 
                                                                  mass_weight=True)
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
        
        # ax.set_xlim(-40,)
        # ax.set_ylim(-30,)

