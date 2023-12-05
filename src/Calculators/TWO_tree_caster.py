#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gives ray which are logspaced. Around photosphere they sould be 1.
Created on Tue Oct 10 10:19:34 2023

@author: paola, konstantinos

"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
alice = False
#%% Constants & Converter
G = 6.6743e-11 # SI
Msol = 1.98847e30 # kg
Rsol = 6.957e8 # m
t = np.sqrt(Rsol**3 / (Msol*G )) # Follows from G = 1
c = 3e8 * t/Rsol # simulator units. Need these for the PW potential
c_cgs = 3e10 # [cm/s]
Msol_to_g = 1.989e33 # [g]
Rsol_to_cm = 6.957e10 # [cm]
den_converter = Msol_to_g / Rsol_to_cm**3


def grid_maker(fix, m, check , x_num, y_num, z_num = 1000):
    """ Outputs are in in solar units """
    fix = str(fix)
    Mbh = 10**m 
    Rt =  Mbh**(1/3) # Msol = 1, Rsol = 1
    sim = str(m) + '-' + check

    if alice:
        pre = '/home/s3745597/data1/TDE/'
        # Import
        X = np.load(pre + sim + '/snap_'  + fix + '/CMx_' + fix + '.npy')
        Y = np.load(pre + sim + '/snap_'  + fix + '/CMy_' + fix + '.npy')
        Z = np.load(pre + sim + '/snap_'  + fix + '/CMz_' + fix + '.npy')
        T = np.load(pre + sim + '/snap_'  + fix + '/T_' + fix + '.npy')
        Den = np.load(pre + sim + '/snap_'  + fix + '/Den_' + fix + '.npy')
        Rad = np.load(pre + sim + '/snap_'  +fix + '/Rad_' + fix + '.npy')
    else:
        # Import
        X = np.load( str(m) + '/'  + fix + '/CMx_' + fix + '.npy')
        Y = np.load( str(m) + '/'  + fix + '/CMy_' + fix + '.npy')
        Z = np.load( str(m) + '/'  + fix + '/CMz_' + fix + '.npy')
        Mass = np.load( str(m) + '/'  + fix + '/Mass_' + fix + '.npy')
        Den = np.load( str(m) + '/'  + fix + '/Den_' + fix + '.npy')
    
    # Convert Energy / Mass to Energy Density in CGS
    Den *= den_converter 
    
    # make a tree
    sim_value = [X, Y, Z] 
    sim_value = np.transpose(sim_value) #array of dim (number_points, 3)
    sim_tree = KDTree(sim_value) 
    
    # Ensure that the regular grid cells are smaller than simulation cells
    x_start = -15_000
    x_stop = 140
    y_start = -4000
    y_stop = 4000
    z_start = -100
    z_stop = 100
    # r_radii = np.logspace(np.log10(x_start), np.log10(x_stop), x_num) #simulator units
    # thetas = np.logspace(np.log10(y_start), np.log10(y_stop), y_num) #simulator units
    # phis = np.logspace(np.log10(z_start), np.log10(z_stop), z_num) #simulator units
    
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
                gridded_den[i, j, k] = Den[idx]
                gridded_mass[i,j, k] = Mass[idx]
        
        # Progress Check
        progress = int(100 * i/len(x_radii))
        if progress != last_progress:
            last_progress = progress    
            print('Progress: {:1.0%}'.format(i/len(x_radii)))
        
    return gridded_indexes, gridded_den, gridded_mass, x_radii, y_radii, z_radii

 
if __name__ == '__main__':
    m = 6
    check = 'fid'
    gridded_indexes, gridded_den, gridded_mass, x_radii, y_radii, z_radii = grid_maker(844, m, check, 100, 100)
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
        
        den_plot = np.nan_to_num(gridded_den, nan = -1, neginf = -1)
        den_plot = np.log10(den_plot[:,:, len(z_radii)//2])
        den_plot = np.nan_to_num(den_plot, nan = 0, neginf= 0)
        print(den_plot)
        # den_plot[den_plot < 0.1] = 0
        
        ax.set_xlabel(r' X [R$_\odot$]', fontsize = 14)
        ax.set_ylabel(r' Y [R$_\odot$]', fontsize = 14)
        img = ax.pcolormesh(x_radii, y_radii, den_plot.T, cmap = 'jet')
        cb = plt.colorbar(img)
        cb.set_label(r'Density [g/cm$^2$]', fontsize = 14)
        ax.set_title('Midplane', fontsize = 16)
    