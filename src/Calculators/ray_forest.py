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
AEK = '#F1C410'
alice = False
#%% Constants & Converter
NSIDE = 4
G = 6.6743e-11 # SI
Msol = 1.98847e30 # kg
Rsol = 6.957e8 # m
t = np.sqrt(Rsol**3 / (Msol*G )) # Follows from G = 1
c = 3e8 * t/Rsol # simulator units. Need these for the PW potential
c_cgs = 3e10 # [cm/s]
Msol_to_g = 1.989e33 # [g]
Rsol_to_cm = 6.957e10 # [cm]
den_converter = Msol_to_g / Rsol_to_cm**2
en_den_converter = Msol_to_g / (Rsol_to_cm  * t**2 ) # Energy Density converter

fix = 844
m = 6
num = 1000

def isalice():
    return alice

def ray_maker(fix, m, check):
    """ Outputs are in in solar units """
    fix = str(fix)
    Mbh = 10**m 
    Rt =  Mbh**(1/3) # Msol = 1, Rsol = 1
    sim = str(m) + '-' + check

    if alice:
        pre = '/home/s3745597/data1/TDE/'
        # Import
        X = np.load(pre + sim + '/snap_'  + fix + '/CMx_' + fix + '.npy') - Rt
        Y = np.load(pre + sim + '/snap_'  + fix + '/CMy_' + fix + '.npy')
        Z = np.load(pre + sim + '/snap_'  + fix + '/CMz_' + fix + '.npy')
        T = np.load(pre + sim + '/snap_'  + fix + '/T_' + fix + '.npy')
        Den = np.load(pre + sim + '/snap_'  + fix + '/Den_' + fix + '.npy')
        Rad = np.load(pre + sim + '/snap_'  +fix + '/Rad_' + fix + '.npy')
        Vol = np.load(pre + sim + '/snap_' + fix + '/Vol_' + fix + '.npy')
    else:
        # Import
        X = np.load( str(m) + '/'  + fix + '/CMx_' + fix + '.npy')
        Y = np.load( str(m) + '/'  + fix + '/CMy_' + fix + '.npy')
        Z = np.load( str(m) + '/'  + fix + '/CMz_' + fix + '.npy')
        Mass = np.load( str(m) + '/'  + fix + '/Mass_' + fix + '.npy')
        Den = np.load( str(m) + '/'  + fix + '/Den_' + fix + '.npy')
        Rad = np.load( str(m) + '/'  +fix + '/Rad_' + fix + '.npy')
    
    # Move pericenter to 0
    X -= Rt
    # Convert Energy / Mass to Energy Density in CGS
    Rad *= Den 
    Rad *= en_den_converter
    Den *= den_converter 
    
    # make a tree
    sim_value = [X, Y, Z] 
    sim_value = np.transpose(sim_value) #array of dim (number_points, 3)
    sim_tree = KDTree(sim_value) 
    
    # Ensure that the regular grid cells are smaller than simulation cells
    x_start = -15_000 
    x_stop = 800
    x_num = 800
    y_start = -1000
    y_stop = 1000
    y_num = 500
    z_start = -150
    z_stop = 150
    z_num = 10
    
    x_radii = np.linspace(x_start, x_stop, x_num) #simulator units
    y_radii = np.linspace(y_start, y_stop, y_num) #simulator units
    z_radii = np.linspace(z_start, z_stop, z_num) #simulator units

    gridded_indexes =  np.zeros(( len(x_radii), len(y_radii), len(z_radii) ))
    gridded_den =  np.zeros(( len(x_radii), len(y_radii), len(z_radii) ))
    gridded_mass =  np.zeros(( len(x_radii), len(y_radii), len(z_radii) ))
    for i in range(len(x_radii)):
        for j in range(len(y_radii)):
            for k in range(len(z_radii)):
                queried_value = [x_radii[i], y_radii[j], z_radii[k]]
                _, idx = sim_tree.query(queried_value)
                                    
                # Store
                gridded_indexes[i, j, k] = idx
                gridded_den[i, j, k] = Den[idx]
                gridded_mass[i,j, k] = Mass[idx]
        
    return gridded_indexes, gridded_den, gridded_mass, x_radii, y_radii, z_radii

 
if __name__ == '__main__':
    m = 6
    gridded_indexes, gridded_den, gridded_mass, x_radii, y_radii, z_radii = ray_maker(844, m, num)
#%% Plot
    plot = False
    if plot:
        import colorcet
        fig, ax = plt.subplots(1,1)
        plt.rcParams['text.usetex'] = True
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['figure.figsize'] = [6, 4]
        plt.rcParams['axes.facecolor']= 	'whitesmoke'
        
        den_plot = np.log10(gridded_den[:,:,0])
        den_plot = np.nan_to_num(den_plot, neginf= -19)
     
        ax.set_xlabel(r' X [R$_\odot$]', fontsize = 14)
        ax.set_ylabel(r' Y [R$_\odot$]', fontsize = 14)
        img = ax.pcolormesh(x_radii, y_radii, den_plot, cmap = 'cet_fire',
                            vmin = 0, vmax = 5)
        cb = plt.colorbar(img)
        cb.set_label(r'Density [g/cm$^2$]', fontsize = 14)
        ax.set_title('Midplane', fontsize = 16)
    