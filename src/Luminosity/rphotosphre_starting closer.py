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
import scipy.integrate as sci
from scipy.interpolate import griddata
import matlab.engine
from sklearn.neighbors import KDTree
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm

from src.Opacity.LTE_loader import T_opac_ex, Rho_opac_ex, rossland_ex, plank_ex, scattering_ex
import src.Utilities.prelude as c
from src.Utilities.parser import parse
from src.Utilities.isalice import isalice
from src.Utilites.loaders import local_loader
alice, plot = isalice()

save = True # WATCH OUT    WATCH OUT    WATCH OUT    WATCH OUT    WATCH OUT
### Load data ---

def es(rho):
   X = 0.90823
   return 0.2 * (1+X) / rho # [1/cm]
    
def ff(rho, T):
    return 0.64e23 * rho * T**(-3.5) / rho # [1/cm] 
if alice:
    realpre = '/home/kilmetisk/data1/TDE/'
    args = parse()
    sim = args.name
    pre = f'{realpre}{sim}/snap_'
    mstar = args.mass
    rstar = args.radius
    Mbh = args.blackhole
    m = int(np.log10(float(Mbh)))
    fixes = np.arange(args.first, args.last + 1)
    single = args.single
    if single:
        fix = args.only
    else:
        raise NameError('You need to set the single flag for this to run \n it is much faster')
else:
    eng = matlab.engine.start_matlab()
    ms = [4, 5, 6]
    ms = [4]
    mstar = 0.5
    rstar = 0.47
fix = 300
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
f_min = c.kb * 1e3 / c.h
f_max = c.kb * 3e13 / c.h
f_num = 1_000
frequencies = np.logspace(np.log10(f_min), np.log10(f_max), f_num)

# Opacity Input
opac_kind = 'LTE'
opac_path = f'src/Opacity/{opac_kind}_data/'
T_cool = np.loadtxt(f'{opac_path}/T.txt')
Rho_cool = np.loadtxt(f'{opac_path}/rho.txt')
plank = np.loadtxt(f'{opac_path}/planck.txt')
rossland = np.loadtxt(f'{opac_path}/ross.txt')
T_cool2 = T_opac_ex
Rho_cool2 = Rho_opac_ex
rossland2 = rossland_ex
plank2 = plank_ex
scattering2 = scattering_ex

### Do it --- --- ---
xyz = np.array([X, Y, Z]).T

# Flux
F_photo = np.zeros((c.NPIX, f_num))
F_photo_temp = np.zeros((c.NPIX, f_num))
i = 150  ################################ observer
mu_x = observers_xyz[i][0]
mu_y = observers_xyz[i][1]
mu_z = observers_xyz[i][2]

# Box is for dynamic ray making
if mu_x < 0:
    rmax = box[0] / mu_x
else:
    rmax = box[3] / mu_x
if mu_y < 0:
    rmax = min(rmax, box[1] / mu_y)
else:
    rmax = min(rmax, box[4] / mu_y)

if mu_z < 0:
    rmax = min(rmax, box[2] / mu_z)
else:
    rmax = min(rmax, box[5] / mu_z)

# Make the ray
rmin = -0.25
rmaxes = np.logspace(np.log10(Rt), np.log10(rmax), 20)
Rphs = []
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
    
    # Interpolate ---
    sigma_rossland = eng.interp2(T_cool2,Rho_cool2,rossland2.T # needs T for the new RICH extrapol
                                 ,np.log(t), np.log(d),'linear',0)
    sigma_rossland = np.array(sigma_rossland)[0]
    sigma_rossland = sigma_rossland[sigma_rossland != 1.0] 
    sigma_rossland_eval = np.exp(sigma_rossland) 
    
    
    # Optical Depth --
    r_fuT = np.flipud(r.T)
    kappa_rossland = np.flipud(sigma_rossland_eval) # + np.flipud(sigma_plank_eval)
    los = - np.flipud(sci.cumulative_trapezoid(kappa_rossland, r_fuT, initial = 0)) * c.Rsol_to_cm # dont know what it do but this is the conversion


    b = np.argmin(np.abs(los-2/3))
    Rphs.append(r[b])
#%%
import matplotlib.pyplot as plt
plt.figure(figsize = (3,3))
plt.title(f'BH: {m} Snapshot {fix} Observer {i}')
plt.plot(rmaxes/amin, np.array(Rphs)/amin, '-o', c = 'k' , lw = 0.75, markersize = 2)
plt.xlabel(r'Where we send the ray from [$\alpha_\mathrm{min}$]')
plt.ylabel(r'$R_\mathrm{photo}$ [$\alpha_\mathrm{min}$]')