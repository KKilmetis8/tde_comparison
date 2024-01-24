"""
Created on January 2024
Author: Paola 

Check if the raymaker with dynamical radii gives the same special radii as Elad

"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

import numpy as np
import matplotlib.pyplot as plt
import h5py
import healpy as hp
from src.Calculators.ray_forest import find_sph_coord, ray_maker_forest
from src.Luminosity.special_radii_tree_cloudy import calc_specialr

snap = 882
m = 6
check = 'fid'
num = 1000
NSIDE = 4
filename = f"{m}/{snap}/snap_{snap}.h5"
Rsol_to_cm = 7e10 #6.957e10 # [cm]
Mbh = 10**m 
Rt =  Mbh**(1/3) # Msol = 1, Rsol = 1
apocenter = 2 * Rt * Mbh**(1/3)

plot = 'profile'

box = np.zeros(6)

with h5py.File(filename, 'r') as fileh:
    for i in range(len(box)):
        box[i] = fileh['Box'][i]

# Find the limit of the box
thetas = np.zeros(192)
phis = np.zeros(192) 
observers = []
stops = np.zeros(192) 
for iobs in range(0,192):
    theta, phi = hp.pix2ang(NSIDE, iobs) # theta in [0,pi], phi in [0,2pi]
    thetas[iobs] = theta
    phis[iobs] = phi
    observers.append( (theta, phi) )
    xyz = find_sph_coord(1, theta, phi)

    mu_x = float(xyz[0])
    mu_y = float(xyz[1])
    mu_z = float(xyz[2])

    # Box is for 
    if(mu_x < 0):
        rmax = box[0] / mu_x
    else:
        rmax = box[3] / mu_x
    if(mu_y < 0):
        rmax = min(rmax, box[1] / mu_y)
    else:
        rmax = min(rmax, box[4] / mu_y)
    if(mu_z < 0):
        rmax = min(rmax, box[2] / mu_z)
    else:
        rmax = min(rmax, box[5] / mu_z)

    stops[iobs] = rmax

# Find special redii with dynamical radius
tree_indexes, rays_T, rays_den, rays, _, rays_radii, _, _ = ray_maker_forest(snap, m, check, thetas, phis, stops, num)

if plot == 'spec_radii':
    with h5py.File(f'data/elad/data_{snap}.mat', 'r') as f:
        Elad_photo = f['r_photo'][0]
        Elad_therm = f['r_therm'][0]

    rays_photo = np.zeros(192)
    rays_thermr = np.zeros(192)
    for j in range(len(observers)):
        branch_indexes = tree_indexes[j]
        branch_T = rays_T[j]
        branch_den = rays_den[j]
        radius = rays_radii[j]
        
        _, _, photo, _, _ = calc_specialr(branch_T, branch_den, radius, branch_indexes, select = 'photo')
        _, _, thermr, _, _ = calc_specialr(branch_T, branch_den, radius, branch_indexes, select = 'thermr_plot')
        rays_photo[j] = photo/Rsol_to_cm
        rays_thermr[j] = thermr/Rsol_to_cm

    fig, ax = plt.subplots(1,2, tight_layout = True)
    ax[0].scatter(np.arange(192), Elad_photo, c = 'k', s = 5, label = 'Elad')
    ax[0].scatter(np.arange(192), rays_photo, c = 'orange', s = 5, label = 'us')
    ax[0].set_xlabel('Observers')
    ax[0].set_ylabel(r'$\log_{10}R_{th} [R_\odot]$')
    ax[0].set_yscale('log')
    ax[0].grid()

    ax[1].scatter(np.arange(192), Elad_therm, c = 'k', s = 5, label = 'Elad')
    ax[1].scatter(np.arange(192), rays_thermr, c = 'orange', s = 5, label = 'us')
    ax[1].set_xlabel('Observers')
    ax[1].set_ylabel(r'$\log_{10}R_{th} [R_\odot]$')
    ax[1].set_yscale('log')
    ax[1].grid()

    plt.legend(fontsize = 10)
    plt.savefig(f'Figs/comparison_special_radii{snap}.png')
    plt.show() 

if plot == 'profile':
    fig, ax = plt.subplots()
    selected_indexes = [80]
    for i in selected_indexes:
        radius = np.delete(rays_radii[i],-1)/Rsol_to_cm
        img = ax.scatter(radius, rays_den[i], c = np.log10(rays_T[i]), marker = '_', label = f'observer {i}')
    cbar = fig.colorbar(img)
    cbar.set_label(r'$\log_{10}T [K]$')
    ax.set_xlabel(r'$\log_{10}R [R_\odot]$')
    ax.set_ylabel(r'$\log_{10}\rho [g/cm^3]$')
    plt.xlim(10, 2*apocenter)
    plt.loglog()
    plt.legend()
    plt.show()
