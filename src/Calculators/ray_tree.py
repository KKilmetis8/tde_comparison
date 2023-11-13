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

def select_observer(angles, angle):
    """ Given an array of angles of lenght = len(observers), 
    gives you the index of the observer at theta/phi = angle. """
    index = np.argmin(np.abs(angles - angle))
    return index

def ray_maker(fix, m, select = False):
    """ Outputs are in CGS """
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
        Mass = np.load( str(m) + '/'  + fix + '/Mass_' + fix + '.npy')
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

    #make a tree
    sim_value = [R, THETA, PHI]
    sim_value = np.transpose(sim_value) 
    sim_tree = KDTree(sim_value) 
    
    # Ensure that the regular grid cells are smaller than simulation cells
    start = 50 #50 Solar radii
    stop = 7_000 
    log_start = np.log10(start)
    log_stop = np.log10(stop)
    log_radii = np.linspace(log_start, log_stop, 1000) #simulator units
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
    for j in range (len(observers)):
        branch_indexes = np.zeros(len(radii))
        branch_T = np.zeros(len(radii))
        branch_den = np.zeros(len(radii))
        branch_energy = np.zeros(len(radii))
        for i,radius in enumerate(radii):
            queried_value = [radius, thetas[j], phis[j]]
            _, idx = sim_tree.query(queried_value)
            branch_indexes[i] = idx
            branch_energy[i] = Rad[idx]
            branch_den[i] = Den[idx]
            branch_T[i] = T[idx]
        branch_energy = np.nan_to_num(branch_energy, neginf = 0)
        branch_den = np.nan_to_num(branch_den, neginf = 0)
        branch_T = np.nan_to_num(branch_T, neginf = 0)
        tree_indexes.append(branch_indexes)
        rays.append(branch_energy)
        rays_den.append(branch_den)
        rays_T.append(branch_T)
    
    if select == True:
        return tree_indexes, rays_T, rays_den, rays, radii, thetas, phis
    
    else:
        return tree_indexes, rays_T, rays_den, rays, radii

 

if __name__ == '__main__':
    m = 6
    tree_indexes, rays_T, rays_den, rays, radii = ray_maker(844, m)
    print(len(rays_T[90]))