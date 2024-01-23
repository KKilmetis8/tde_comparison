#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test if the bolometric luminosity from the single observers gives you the same value as red

@author: paola 

"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

from src.Utilities.isalice import isalice
alice, plot = isalice()
Rsol_to_cm = 7e10 #6.957e10 # [cm]

# Vanilla imports
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.stats import gmean
from src.Luminosity.special_radii_tree_cloudy import calc_specialr
from src.Luminosity.select_path import select_snap
from src.Calculators.ray_forest import find_sph_coord, ray_maker
import h5py
import healpy as hp

plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [5 , 4]
pre = '/home/s3745597/data1/TDE/tde_comparison'

m = 6
#snap = 882
check = 'fid'
num = 1000
NSIDE = 4

# nL_tilde_n = np.loadtxt(f'data/blue/TESTnorm_nLn_single_m6_881.txt')
# nL_tilde_n_new = np.zeros(len(nL_tilde_n[1]))
# for i in range(len(nL_tilde_n[1])):
#     for iobs in range(len(nL_tilde_n_new)):
#         nL_tilde_n_new[i] += nL_tilde_n[iobs][i]
# #nL_tilde_n_new /= 192
# x_array = np.loadtxt(f'data/blue/spectrafreq_m{m}.txt')
# n_array = np.power(10, x_array)

# Ltot = 0 
# for i in range(len(nL_tilde_n)):
#     xLx =  n_array * nL_tilde_n[i]
#     L = np.trapz(xLx, x_array) 
#     L *= np.log(10)
#     Ltot += L
# Ltot /= len(nL_tilde_n)

# print(Ltot)

# L = np.trapz(n_array * nL_tilde_n_new, x_array) 
# L *= np.log(10)
# L /= len(nL_tilde_n)
# print(L)

# snapshots = np.arange(844,1008+1)
# amean_rtherm = np.zeros(len(snapshots))
# gmean_rtherm = np.zeros(len(snapshots))
# for idx_sn, snap in enumerate(snapshots):
#     with h5py.File(f'data/elad/data_{snap}.mat', 'r') as f:
#         rtherm = f['r_photo'][0]
#     amean_rtherm[idx_sn] = np.mean(rtherm)
#     gmean_rtherm[idx_sn] = gmean(rtherm)
# with open(f'data/elad_photo.txt', 'a') as file:  
#     file.write(' '.join(map(str, amean_rtherm)) + '\n')
#     file.write(' '.join(map(str, gmean_rtherm)) + '\n')
#     file.close()

snapshots, days = select_snap(m, check)
fix_photo_arit = np.zeros(len(snapshots))
fix_photo_geom = np.zeros(len(snapshots))
fix_thermr_arit = np.zeros(len(snapshots))
fix_thermr_geom = np.zeros(len(snapshots))

for index in range(1,2):#len(snapshots)): 
    snap = snapshots[index]
    print(snap)       
    box = np.zeros(6)
    filename = f"{m}/{snap}/snap_{snap}.h5"
    
    with h5py.File(filename, 'r') as fileh:
        for i in range(len(box)):
            box[i] = fileh['Box'][i]

    with h5py.File(f'data/elad/data_{snap}.mat', 'r') as f:
        Elad_photo = f['r_photo'][0]
        Elad_therm = f['r_therm'][0]

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

        mu_x = xyz[0]
        mu_y = xyz[1]
        mu_z = xyz[2]

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

    # rays is Er of Elad 
    tree_indexes, rays_T, rays_den, rays, rays_ie, rays_radii, _, rays_v = ray_maker(snap, m, check, thetas, phis, stops, num)
    
    rays_photo = np.zeros(192)
    rays_thermr = np.zeros(192)
    for j in range(len(observers)):
        branch_indexes = tree_indexes[j]
        branch_T = rays_T[j]
        branch_den = rays_den[j]
        radius = rays_radii[j]
        
        _, _, photo, _, _ = calc_specialr(branch_T, branch_den, radius, branch_indexes, select = 'photo')
        _, _, thermr, _, _ = calc_specialr(branch_T, branch_den, radius, branch_indexes, select = 'thermr_plot')
        rays_photo[j] = photo
        rays_thermr[j] = thermr

    plt.figure()
    plt.scatter(range(192), rays_photo/Rsol_to_cm, c = 'r', s = 12, label = 'us')
    plt.scatter(range(192), Elad_photo, c = 'k', s = 8, label = 'Elad')
    plt.legend()
    plt.savefig(f'Figs/photocomp{snap}.png')
    plt.show()

#     fix_photo_arit[index] = np.mean(rays_photo)/Rsol_to_cm
#     fix_photo_geom[index] = gmean(rays_photo)/Rsol_to_cm
#     fix_thermr_arit[index] = np.mean(rays_thermr)/Rsol_to_cm
#     fix_thermr_geom[index] = gmean(rays_thermr)/Rsol_to_cm

#     pre_saving = 'data/'

# with open(f'{pre_saving}TESTspecial_radii_m{m}_oldopacity.txt', 'a') as file:
#     file.write('# Run of' + '\n#t/t_fb\n')
#     file.write(' '.join(map(str, days)) + '\n')
#     file.write('# Photosphere arithmetic mean \n')
#     file.write(' '.join(map(str, fix_photo_arit)) + '\n')
#     file.write('# Photosphere geometric mean \n')
#     file.write(' '.join(map(str, fix_photo_geom)) + '\n')
#     file.write('# Thermalisation radius arithmetic mean \n')
#     file.write(' '.join(map(str, fix_thermr_arit)) + '\n')
#     file.write('# Thermalisation radius geometric mean \n')
#     file.write(' '.join(map(str, fix_thermr_geom)) + '\n')
#     file.close()
