#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:55:33 2025

@author: konstantinos
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 16:48:31 2025

@author: konstantinos
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:30:18 2025

@author: konstantinos
"""

import os
import gc
import time
import warnings
warnings.filterwarnings('ignore')
import csv
import copy

# Vanilla imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import healpy as hp
import scipy.integrate as sci
from scipy.interpolate import griddata
import matlab.engine
from sklearn.neighbors import KDTree
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm

# from src.Opacity.LTE_loader import T_opac_ex, Rho_opac_ex, rossland_ex, plank_ex, scattering_ex
from src.Opacity.linextrapolator import pad_interp, extrapolator_flipper, nouveau_rich

opac_kind = 'LTE'
opac_path = f'src/Opacity/{opac_kind}_data/'
T_cool = np.loadtxt(f'{opac_path}/T.txt')
Rho_cool = np.loadtxt(f'{opac_path}/rho.txt')
plank = np.loadtxt(f'{opac_path}/planck.txt')
rossland = np.loadtxt(f'{opac_path}/ross.txt')
scattering = np.loadtxt(f'{opac_path}/scatter.txt')

import src.Utilities.prelude as c
from src.Utilities.parser import parse
from src.Utilities.isalice import isalice
from src.Utilities.loaders import local_loader, boxer

alice, plot = isalice()

save = True # WATCH OUT    WATCH OUT    WATCH OUT    WATCH OUT    WATCH OUT
### Load data ---

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
    mstar = 0.5
    rstar = 0.47
def masker(mask, list_of_quantities):
    new_list = []
    for quantity in list_of_quantities:
        new_list.append(quantity[mask])
    return (*new_list,)

m = 6
fix = 444
corrections = np.array([-0.5, -1, -1.5, -2, -2.5, -3, -10])
avg_eq_photo = np.zeros_like(corrections)
for icor, cor in tqdm(enumerate(corrections)):
    Rt = rstar * (10**m/mstar)**(1/3)
    try:
        X, Y, Z, Den, T, Rad, Vol, box, day = local_loader(m, fix,
                                                           'thermodynamics')
    except:
        continue
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
    N_ray = 1_000
    
    # Opacity Input
    T_cool2, Rho_cool2, rossland2 = nouveau_rich(T_cool, Rho_cool, rossland, 
                                                  what = 'scattering',
                                                  correction = cor)
    ### Do it --- --- ---
    xyz = np.array([X, Y, Z]).T
    
    # Make the ray
    rmin = -0.25
    Rphs = []
    # observers = np.linspace(0,c.NPIX - 1,5)
    observers = np.arange(88, 104)[::2] # Equatorial
    # observers = [92,]
    x_photo = np.zeros(len(observers))
    y_photo = np.zeros(len(observers))
    #los_es = []
    #rs = []
    for i, obs in tqdm(zip(np.arange(0,len(observers)), observers,)):
            # Make Ray
            obs = int(obs)
            rmax = boxer(obs, observers_xyz, box)
            r = np.logspace(rmin, np.log10(rmax), N_ray)
            
            # Get Ray XYZ
            mu_x = observers_xyz[obs][0]
            mu_y = observers_xyz[obs][1]
            mu_z = observers_xyz[obs][2]
            x = r*mu_x
            y = r*mu_y
            z = r*mu_z
            xyz2 = np.array([x, y, z]).T
            tree = KDTree(xyz, leaf_size=50)
            _, idx = tree.query(xyz2, k=1)
            idx = [ int(idx[j][0]) for j in range(len(idx))] # no -1 because we start from 0
        
            d = Den[idx] * c.den_converter
            t = T[idx]
            ray_x = X[idx]
            ray_y = Y[idx]
            ray_z = Z[idx]
            
            # Interpolate ---
            sigma_rossland = eng.interp2(T_cool2,Rho_cool2,rossland2.T # needs T for the new RICH extrapol
                                         ,np.log(t), np.log(d),'linear',0)
            sigma_rossland = np.array(sigma_rossland)[0]
            underflow_mask = sigma_rossland != 0.0
            d, t, r, sigma_rossland = masker(underflow_mask, [d, t, r, sigma_rossland])
            sigma_rossland_eval = np.exp(sigma_rossland) 
            
            # Optical Depth --
            r_fuT = np.flipud(r)
            kappa_rossland = np.flipud(sigma_rossland_eval)
            los = - np.flipud(sci.cumulative_trapezoid(kappa_rossland, 
                                                       r_fuT, 
                                                       initial = 0)) * c.Rsol_to_cm
            #los_es.append(los)
            photosphere = np.where(los < 2/3)[0][0]
            
            #rs.append(r)
            Rphs.append(r[photosphere])
    avg_eq_photo[icor] = np.mean(Rphs) / amin
#%%    
fig, ax = plt.subplots(1,1, figsize = (3,3))
ax.set_title(f'$10^{m}$ M$_\odot$, snapshot {fix}')
ax.plot(corrections, avg_eq_photo, ':h', c = 'k')
ax.set_xlabel('Slope of power law', fontsize = 12)
ax.set_ylabel(r'Average Equatorial Photosphere [$\alpha_\mathrm{min}$]', 
              fontsize = 12)
#ax.set_xscale('')

    