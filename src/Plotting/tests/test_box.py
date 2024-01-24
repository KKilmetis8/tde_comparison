"""
Test the box computing the photosphere/Rtherm with and without dynamical radius

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
import healpy as hp
from scipy.stats import gmean
from src.Luminosity.special_radii_tree_cloudy import calc_specialr, get_specialr
from src.Luminosity.select_path import select_snap
from src.Calculators.ray_forest import find_sph_coord, ray_maker_forest
from src.Calculators.ray_tree import ray_maker

plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [5 , 4]
NSIDE = 4

m = 6
check = 'fid'
num = 1000

snapshots, days = select_snap(m, check)

fix_photo_arit = np.zeros(len(snapshots))
fix_photo_geom = np.zeros(len(snapshots))
fix_thermr_arit = np.zeros(len(snapshots))
fix_thermr_geom = np.zeros(len(snapshots))

singlefix_photo_arit = np.zeros(len(snapshots))
singlefix_photo_geom = np.zeros(len(snapshots))
singlefix_thermr_arit = np.zeros(len(snapshots))
singlefix_thermr_geom = np.zeros(len(snapshots))


for index in range(len(snapshots)): 
    snap = snapshots[index]
    print(snap)       

    with h5py.File(f'data/elad/data_{snap}.mat', 'r') as f:
        Elad_photo = f['r_photo'][0]
        Elad_therm = f['r_therm'][0]

    # Find the limit of the box
    box = np.zeros(6)
    filename = f"{m}/{snap}/snap_{snap}.h5"
    with h5py.File(filename, 'r') as fileh:
        for i in range(len(box)):
            box[i] = fileh['Box'][i]
    thetas = np.zeros(192)
    phis = np.zeros(192) 
    observers = []
    stops = np.zeros(192) 
    for iobs in range(0,192):
        theta, phi = hp.pix2ang(NSIDE, iobs) # theta in [0,pi], phi in [0, 2pi]
        thetas[iobs] = theta
        phis[iobs] = phi
        observers.append( (theta, phi) )
        xyz = find_sph_coord(1, theta, phi)

        mu_x = xyz[0]
        mu_y = xyz[1]
        mu_z = xyz[2]

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

    # Find special redii with fixed radius
    single_tree_indexes, _, single_rays_T, single_rays_den, single_rays, _, single_radii, _, _ = ray_maker(snap, m, check, num)
    _, _, single_photo, _, _ = get_specialr(single_rays_T, single_rays_den, single_radii, single_tree_indexes, select='photo')
    _, _, single_thermr, _, _ = get_specialr(single_rays_T, single_rays_den, single_radii, single_tree_indexes, select= 'thermr_plot')

    singlefix_photo_arit[index] = np.mean(single_photo)/Rsol_to_cm
    singlefix_photo_geom[index] = gmean(single_photo)/Rsol_to_cm
    singlefix_thermr_arit[index] = np.mean(single_thermr)/Rsol_to_cm
    singlefix_thermr_geom[index] = gmean(single_thermr)/Rsol_to_cm

    # Find special redii with dynamical radius
    tree_indexes, rays_T, rays_den, rays, _, rays_radii, _, _ = ray_maker_forest(snap, m, check, thetas, phis, stops, num)
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

    fix_photo_arit[index] = np.mean(rays_photo)/Rsol_to_cm
    fix_photo_geom[index] = gmean(rays_photo)/Rsol_to_cm
    fix_thermr_arit[index] = np.mean(rays_thermr)/Rsol_to_cm
    fix_thermr_geom[index] = gmean(rays_thermr)/Rsol_to_cm


with open(f'data/TESTspecial_radii_m{m}_box.txt', 'a') as file:
    file.write('# Using dynamical boxes' + '\n#t/t_fb\n')
    file.write(' '.join(map(str, days)) + '\n')
    file.write('# Photosphere arithmetic mean \n')
    file.write(' '.join(map(str, fix_photo_arit)) + '\n')
    file.write('# Photosphere geometric mean \n')
    file.write(' '.join(map(str, fix_photo_geom)) + '\n')
    file.write('# Thermalisation radius arithmetic mean \n')
    file.write(' '.join(map(str, fix_thermr_arit)) + '\n')
    file.write('# Thermalisation radius geometric mean \n')
    file.write(' '.join(map(str, fix_thermr_geom)) + '\n')
    file.write('# Using fix radius \n')
    file.write('# Photosphere arithmetic mean \n')
    file.write(' '.join(map(str, singlefix_photo_arit)) + '\n')
    file.write('# Photosphere geometric mean \n')
    file.write(' '.join(map(str, singlefix_photo_geom)) + '\n')
    file.write('# Thermalisation radius arithmetic mean \n')
    file.write(' '.join(map(str, singlefix_thermr_arit)) + '\n')
    file.write('# Thermalisation radius geometric mean \n')
    file.write(' '.join(map(str, singlefix_thermr_geom)) + '\n')
    file.close()
