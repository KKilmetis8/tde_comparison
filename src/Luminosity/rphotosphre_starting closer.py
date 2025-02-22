#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 17:10:54 2025

@author: konstantinos
"""

import sys
import gc
import time
import warnings
warnings.filterwarnings('ignore')
import csv

# Vanilla imports
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import scipy.integrate as sci
from scipy.interpolate import griddata
import matlab.engine
from sklearn.neighbors import KDTree
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm

from src.Opacity.LTE_loader import T_opac_ex, Rho_opac_ex, rossland_ex, plank_ex, scattering_ex
T_cool2 = T_opac_ex
Rho_cool2 = Rho_opac_ex
rossland2 = rossland_ex
plank2 = plank_ex
scattering2 = scattering_ex
import src.Utilities.prelude as c
from src.Utilities.parser import parse
from src.Utilities.isalice import isalice
from src.Utilities.loaders import local_loader, boxer
alice, plot = isalice()
def masker(mask, list_of_quantities):
    new_list = []
    for quantity in list_of_quantities:
        new_list.append(quantity[mask])
    return (*new_list,)
save = True # WATCH OUT    WATCH OUT    WATCH OUT    WATCH OUT    WATCH OUT
### Load data ---

eng = matlab.engine.start_matlab()
ms = [4, 5, 6]
ms = [4]
mstar = 0.5
rstar = 0.47
fix = 348
m = 4
pre = f'{m}/'
Rt = rstar * (10**m/mstar)**(1/3)

X, Y, Z, Den, T, Rad, Vol, box, day = local_loader(m, fix, 'thermodynamics')
Rad_den = np.multiply(Rad,Den)

R = np.sqrt(X**2 + Y**2 + Z**2)
Rt = rstar * (10**m/mstar)**(1/3)
amin = Rt * (10**m/mstar)**(1/3)

# Cross dot ---
observers_xyz = hp.pix2vec(c.NSIDE, range(c.NPIX))
observers_xyz = np.array([observers_xyz]).T[:,:,0]
# Line 17, * is matrix multiplication, ' is .T
cross_dot = np.matmul(observers_xyz,  observers_xyz.T )
cross_dot[cross_dot<0] = 0
cross_dot *= 4/192

# Freq range
N_ray = 5_000

### Do it --- --- ---
xyz = np.array([X, Y, Z]).T

# Flux
plt.figure(figsize = (3,3))
observers = np.arange(0, c.NPIX, 20)  ################################ observer
#observers = [104, 191]
for i, color in zip(observers, c.r10_palette):
    mu_x = observers_xyz[i][0]
    mu_y = observers_xyz[i][1]
    mu_z = observers_xyz[i][2]
    
    # Make the ray
    # rmax = boxer(i, observers_xyz, box)
    rmin = -0.25
    # Make the ray
    rmaxes = [1e-2 * amin, 1e-1 * amin, amin, 10*amin, 50*amin, 100*amin, 1000*amin]
    Rphs = []
    errors = []
    for rmax in tqdm(rmaxes):
        r = np.logspace(rmin, np.log10(rmax), N_ray)
        # r = np.linspace(0.1*amin, 5*amin, N_ray)
        x = r*mu_x
        y = r*mu_y
        z = r*mu_z
        xyz2 = np.array([x, y, z]).T
        tree = KDTree(xyz, leaf_size=50)
        _, idx = tree.query(xyz2, k=1)
        idx = [ int(idx[i][0]) for i in range(len(idx))] # no -1 because we start from 0
        del x, y, z
    
        d = Den[idx] * c.den_converter
        t = T[idx]
        ray_x = X[idx]
        ray_y = Y[idx]
        ray_z = Z[idx]
        vol = Vol[idx]
        
        # Interpolate ---
        sigma_rossland = eng.interp2(T_cool2,Rho_cool2,rossland2.T # needs T for the new RICH extrapol
                                     ,np.log(t), np.log(d),'linear',0)
        sigma_rossland = np.array(sigma_rossland)[0]
        underflow_mask = sigma_rossland != 1.0
        sigma_rossland, vol = masker(underflow_mask, [sigma_rossland, vol])
        sigma_rossland_eval = np.exp(sigma_rossland) 
        
        # Optical Depth --
        r_fuT = np.flipud(r.T)
        kappa_rossland = np.flipud(sigma_rossland_eval) # + np.flipud(sigma_plank_eval)
        los = - np.flipud(sci.cumulative_trapezoid(kappa_rossland, r_fuT, initial = 0)) * c.Rsol_to_cm # dont know what it do but this is the conversion
    
        
        b = np.argmin(np.abs(los-2/3))
        Rphs.append(r[b])
        
        cell_radius = vol**(1/3)
        errors.append(cell_radius[b])
    plt.errorbar(np.array(rmaxes)/amin, np.array(Rphs)/amin, ls = '-', 
                 ecolor = 'k', marker = 'o',
             c = color, yerr = np.array(errors)/amin, capsize = 2,
             lw = 0.75, markersize = 1, label = f'{i}')
plt.loglog()
plt.title(f'BH: {m} Snapshot {fix}')
plt.xlabel(r'Where we send the ray from [$\alpha_\mathrm{min}$]')
plt.ylabel(r'$R_\mathrm{photo}$ [$\alpha_\mathrm{min}$]')
plt.legend(frameon = False, fontsize = 7, ncols = 2)