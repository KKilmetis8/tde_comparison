"""
Created on January 2024
Author: Paola 

Investigate if the arithmentic mean is correct

"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.stats import gmean
from src.Calculators.ray_tree import ray_maker
from src.Luminosity.special_radii_tree_cloudy import get_specialr

snap = 844
m = 6
check = 'fid'
num = 1000

with h5py.File(f'data/elad/data_{snap}.mat', 'r') as f:
    rtherm = f['r_photo'][0]
    amean_rtherm = np.mean(rtherm)
    gmean_rtherm = gmean(rtherm)

rtherm /= amean_rtherm

tree_indexes, observers, rays_T, rays_den, _, radii, _ = ray_maker(snap, m, check, num)
_, rays_cumulative_taus, specialr, _, _ = get_specialr(rays_T, rays_den, radii, tree_indexes, select = 'thermr')
our_mean = np.mean(specialr)
our_rtherm = specialr / our_mean

plt.figure()
plt.scatter(np.arange(192), rtherm, c = 'k', s = 10, label = 'Elad')
plt.scatter(np.arange(192), our_rtherm, c = 'b', s = 8, label = 'us')
plt.xlabel('Observers')
plt.ylabel(r'$R_{therm}/\bar{R}_{therm}$')
plt.ylim(0,10)
plt.legend()
plt.show()