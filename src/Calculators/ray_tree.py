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
import healpy as hp
from src.Luminosity.select_path import select_prefix
import matplotlib.pyplot as plt
AEK = '#F1C410'
from src.Utilities.isalice import isalice
alice, plot = isalice()
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
fix = 844
m = 6
num = 1000

def isalice():
    return alice

def find_sph_coord(r, theta,phi):
    x = r * np.sin(np.pi-theta) * np.cos(phi) #because theta should start from the z axis: we're flipped
    y = r * np.sin(np.pi-theta) * np.sin(phi)
    z = r * np.cos(np.pi-theta)
    return [x,y,z]

def ray_maker(fix, m, check, num = 1001): 
    """ 
    Num is 1001 because for blue we then delete the last cell.
    Outputs are in CGS with exception of ray_vol (in solar units).
    """
    fix = str(fix)
    Mbh = 10**m 
    Rt =  Mbh**(1/3) # Msol = 1, Rsol = 1
    apocenter = 2 * Rt * Mbh**(1/3)

    # Load data
    pre = select_prefix(m, check)
    X = np.load(pre + fix + '/CMx_' + fix + '.npy')
    Y = np.load(pre + fix + '/CMy_' + fix + '.npy')
    Z = np.load(pre + fix + '/CMz_' + fix + '.npy')
    T = np.load(pre + fix + '/T_' + fix + '.npy')
    Den = np.load(pre + fix + '/Den_' + fix + '.npy')
    Rad = np.load(pre + fix + '/Rad_' + fix + '.npy')
    Vol = np.load(pre + fix + '/Vol_' + fix + '.npy')
    
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
    start = 0.1 * Rt #Solar radii
    stop = apocenter #apocenter for 10^6 is 20_000 
    log_start = np.log10(start)
    log_stop = np.log10(stop)
    log_radii = np.linspace(log_start, log_stop, num) #simulator units
    radii = 10**log_radii
    
    # Find observers with Healpix
    thetas = np.zeros(192)
    phis = np.zeros(192) 
    observers = []
    for i in range(0,192):
        thetas[i], phis[i] = hp.pix2ang(NSIDE, i) # theta in [0,pi], phi in [0,2pi]
        #phis[i] -= np.pi # Enforce theta in -pi to pi for astropy
        observers.append( (thetas[i], phis[i]) )
    
    tree_indexes = np.zeros((len(observers), len(radii)))
    rays_T = np.zeros((len(observers), len(radii)))
    rays_den = np.zeros((len(observers), len(radii)))
    rays = np.zeros((len(observers), len(radii)))
    rays_vol = np.zeros((len(observers), len(radii)))
    for j in range(len(observers)):
        for k, radius in enumerate(radii):
            queried_value = find_sph_coord(radius, thetas[j], phis[j])
            _, idx = sim_tree.query(queried_value)

            # Store
            tree_indexes[j][k] = idx 
            rays_T[j][k] = T[idx] 
            rays_den[j][k] = Den[idx] 
            rays[j][k] = Rad[idx] 
            rays_vol[j][k] = Vol[idx] # not in CGS

    # Remove Bullshit
    rays = np.nan_to_num(rays, neginf = 0)
    rays_den = np.nan_to_num(rays_den, neginf = 0)
    rays_T = np.nan_to_num(rays_T, neginf = 0)

    #%% Plot
    # ax = plt.figure().add_subplot(projection='3d')
    # radii = np.array(radii)
    # ax.scatter(our_x[::10], our_y[::10], our_z[::10], 
    #             c = radii[::10], cmap = 'cet_bmy', alpha = 0.18, zorder = 2)
    # # Selecting one ray
    # pick = 107
    # rat = pick * num
    # old_rat = rat - num
    # r_rat = np.sqrt(our_x[old_rat:rat]**2 + our_y[old_rat:rat]**2 + our_z[old_rat:rat]**2)
    # ax.scatter(our_x[old_rat:rat], our_y[old_rat:rat], our_z[old_rat:rat], 
    #             c = 'k', alpha = 1, zorder = 10)
    # ax.set_xlim(-10_000, 10_000)
    # ax.set_ylim(-10_000, 10_000)
    # ax.set_zlim(-10_000, 10_000)
    
        
    # Convert to CGS
    radii *= Rsol_to_cm
    
    return tree_indexes, observers, rays_T, rays_den, rays, radii, rays_vol

 
if __name__ == '__main__':
    m = 6
    num = 1000
    tree_indexes, _, rays_T, rays_den, rays, radii, rays_vol = ray_maker(844, m, num)
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
        
        den_plot = np.log10(rays_den)
        den_plot = np.nan_to_num(den_plot, neginf= -19)
        den_plot = np.reshape(den_plot, (192, len(radii)))
        ax.set_ylabel('Observers', fontsize = 14)
        ax.set_xlabel(r'r [R$_\odot$]', fontsize = 14)
        img = ax.pcolormesh(radii/Rsol_to_cm, range(len(rays_den)), den_plot, cmap = 'cet_fire',
                            vmin = -17, vmax = - 7)
        cb = plt.colorbar(img)
        cb.set_label(r'Density [g/cm$^3$]', fontsize = 14)
        ax.set_title('N: ' + str(num), fontsize = 16)
    