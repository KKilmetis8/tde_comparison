#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gives ray which are logspaced. Around photosphere they sould be 1.
Created on Tue Oct 10 10:19:34 2023

@author: paola

"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

import numpy as np
from scipy.spatial import KDTree
import healpy as hp
from astropy.coordinates import cartesian_to_spherical
import matplotlib.pyplot as plt
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
den_converter = Msol_to_g / Rsol_to_cm**3
en_den_converter = Msol_to_g / (Rsol_to_cm  * t**2 ) # Energy Density converter

def ray_maker(fix, m, num = 5000):
    """ Outputs are in CGS with exception of ray_vol (in solar units) """
    fix = str(fix)
    Mbh = 10**m 
    Rt =  Mbh**(1/3) # Msol = 1, Rsol = 1
    
    if alice:
        pre = '/home/s3745597/data1/TDE/'
        # Import
        X = np.load(pre + str(m) + '/snap_'  + fix + '/CMx_' + fix + '.npy')
        Y = np.load(pre + str(m) + '/snap_'  + fix + '/CMy_' + fix + '.npy')
        Z = np.load(pre + str(m) + '/snap_'  + fix + '/CMz_' + fix + '.npy')
        Mass = np.load(pre + str(m) + '/snap_'  + fix + '/Mass__' + fix + '.npy')
        T = np.load(pre + str(m) + '/snap_'  + fix + '/T_' + fix + '.npy')
        Den = np.load(pre + str(m) + '/snap_'  + fix + '/Den_' + fix + '.npy')
        Rad = np.load(pre + str(m) + '/snap_'  +fix + '/Rad_' + fix + '.npy')
    else:
        # Import
        X = np.load( str(m) + '/'  + fix + '/CMx_' + fix + '.npy')
        Y = np.load( str(m) + '/'  + fix + '/CMy_' + fix + '.npy')
        Z = np.load( str(m) + '/'  + fix + '/CMz_' + fix + '.npy')
        Vol = np.load( str(m) + '/'  + fix + '/Vol_' + fix + '.npy')
        T = np.load( str(m) + '/'  + fix + '/T_' + fix + '.npy')
        Den = np.load( str(m) + '/'  + fix + '/Den_' + fix + '.npy')
        Rad = np.load( str(m) + '/'  +fix + '/Rad_' + fix + '.npy')

    # Convert Energy / Mass to Energy Density in CGS
    Rad *= Den 
    Rad *= en_den_converter
    Den *= den_converter 
    R, THETA, PHI = cartesian_to_spherical(X,Y,Z)
    R = R.value 
    THETA = THETA.value
    PHI = PHI.value

    # make a tree
    sim_value = [R, THETA, PHI] 
    sim_value = np.transpose(sim_value) #array of dim (number_points, 3)
    sim_tree = KDTree(sim_value) 
    
    # Ensure that the regular grid cells are smaller than simulation cells
    start = 0.5 * Rt #Solar radii
    stop = 10_000 
    log_start = np.log10(start)
    log_stop = np.log10(stop)
    log_radii = np.linspace(log_start, log_stop, num) #simulator units
    radii = 10**log_radii
    
    # Find observers with Healpix
    thetas = np.zeros(192)
    phis = np.zeros(192)
    observers = []
    for i in range(0,192):
        thetas[i], phis[i] = hp.pix2ang(NSIDE, i)
        thetas[i] -= np.pi/2 # Enforce theta in -pi to pi
        observers.append( (thetas[i], phis[i]) )

    #%% Find the neighbour to the cell of our grid and keep its values
    tree_indexes = []
    rays_T = []
    rays_den = []
    rays = []
    rays_vol = []
    for j in range(len(observers)):
        branch_indexes = np.zeros(len(radii))
        branch_T = np.zeros(len(radii))
        branch_den = np.zeros(len(radii))
        branch_energy = np.zeros(len(radii))
        branch_vol = np.zeros(len(radii)) # not in CGS
        for i,radius in enumerate(radii):
            queried_value = [radius, thetas[j], phis[j]]
            _, idx = sim_tree.query(queried_value)
            branch_indexes[i] = idx
            branch_T[i] = T[idx]
            branch_den[i] = Den[idx]
            branch_energy[i] = Rad[idx]
            branch_vol[i] = Vol[idx] # not in CGS
        
        # Remove Bullshit
        branch_energy = np.nan_to_num(branch_energy, neginf = 0)
        branch_den = np.nan_to_num(branch_den, neginf = 0)
        branch_T = np.nan_to_num(branch_T, neginf = 0)
        
        # Store as rays
        tree_indexes.append(branch_indexes)
        rays_T.append(branch_T)
        rays_den.append(branch_den)
        rays.append(branch_energy)
        rays_vol.append(branch_vol) # not in CGS
        
    # Convert to CGS
    radii *= Rsol_to_cm

    return tree_indexes, rays_T, rays_den, rays, radii, rays_vol

 
if __name__ == '__main__':
    m = 6
    tree_indexes, rays_T, rays_den, rays, radii, rays_vol = ray_maker(844, m)
    #%%
    fig, ax = plt.subplots(1,1)
    import colorcet
    plt.rcParams['text.usetex'] = True
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['figure.figsize'] = [6, 4]
    plt.rcParams['axes.facecolor']= 	'whitesmoke'
    
    den_plot = np.log10(rays_den)
    den_plot = np.nan_to_num(den_plot, neginf= -19)
    den_plot = np.reshape(den_plot, (192, len(radii)))
    ax.set_ylabel('Observers', fontsize = 14)
    ax.set_xlabel(r'r [R$_\odot$]', fontsize = 14)
    img = ax.pcolormesh(radii/Rsol_to_cm, range(len(rays_den)), den_plot, cmap = 'cet_fire',
                        vmin = -17, vmax = - 7)
    cb = plt.colorbar(img)
    cb.set_label(r'Density [g/cm$^3$]', fontsize = 14)
    ax.set_title('N: ' + str(cell_num), fontsize = 16)
    