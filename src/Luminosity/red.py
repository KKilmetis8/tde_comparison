#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 12:56:40 2025

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
eng = matlab.engine.start_matlab()
from sklearn.neighbors import KDTree
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm

# Chocolate
from src.Opacity.LTE_loader import T_opac_ex, Rho_opac_ex, rossland_ex, plank_ex, scattering_ex
# Opacity Input
T_cool2 = T_opac_ex
Rho_cool2 = Rho_opac_ex
rossland2 = rossland_ex
plank2 = plank_ex
scattering2 = scattering_ex
        
import src.Utilities.prelude as c
from src.Utilities.parser import parse
from src.Utilities.isalice import isalice
from src.Utilities.loaders import local_loader, boxer, alice_loader

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
        fixes = [args.only]
    else:
        raise NameError('You need to set the single flag for this to run \n it is much faster')
else:
    m = 4
    mstar = 0.5
    rstar = 0.47
    if m == 4:
        fixes = [116, 136, 164, 179, 199, 218, 240, 272, 297, 300, 348]
        fixes = [348]+
    if m == 5:
        fixes = [227, 236, 288, 301, 308, 349]
        fixes = [349]
    if m == 6:
        fixes = [180, 290, 315, 325, 351, 379, 444]
        fixes = [444]
    
def masker(mask, list_of_quantities):
    new_list = []
    for quantity in list_of_quantities:
        new_list.append(quantity[mask])
    return (*new_list,)

        
Rt = rstar * (10**m/mstar)**(1/3)
amin = Rt * (10**m/mstar)**(1/3)
    
for fix in fixes:
    try:
        if alice:
            X, Y, Z, Den, T, Rad, Vol, box, day = alice_loader(sim, fix,
                                                               'thermodynamics')
        else:
            X, Y, Z, Den, T, Rad, Vol, box, day = local_loader(m, fix,
                                                               'thermodynamics')
    except:
        continue
    Rad_den = np.multiply(Rad,Den)
    R = np.sqrt(X**2 + Y**2 + Z**2)
    
    # Observers initialization ---
    observers_xyz = hp.pix2vec(c.NSIDE, range(c.NPIX))
    observers_xyz = np.array([observers_xyz]).T[:,:,0]

    # Freq range
    reds = np.zeros(c.NPIX)
    N_ray = 5_000
    observers = np.arange(0,c.NPIX)

    ### Do it --- --- ---
    xyz = np.array([X, Y, Z]).T
    rmin = -0.25
    for i, obs in tqdm(enumerate(observers)):
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
        rad_den = Rad_den[idx]
    
        # Interpolate ---
        sigma_rossland = eng.interp2(T_cool2,Rho_cool2,rossland2.T # needs T for the new RICH extrapol
                                     ,np.log(t), np.log(d),'linear',0)
        sigma_rossland = np.array(sigma_rossland)[0]
        underflow_mask = sigma_rossland != 0.0
        d, t, r, sigma_rossland, ray_x, ray_y, ray_z, rad_den = masker(underflow_mask, 
                                                              [d, t, r, 
                                                               sigma_rossland, 
                                                               ray_x, ray_y, ray_z,
                                                               rad_den])
        sigma_rossland_eval = np.exp(sigma_rossland) 
        
        # Optical Depth --
        r_fuT = np.flipud(r)
        kappa_rossland = np.flipud(sigma_rossland_eval)
        los = - np.flipud(sci.cumulative_trapezoid(kappa_rossland, 
                                                   r_fuT, 
                                                   initial = 0)) * c.Rsol_to_cm
        
        # Red ---
        # Get 20 unique, nearest neighbors
        xyz3 = np.array([ray_x, ray_y, ray_z]).T
        _, idxnew = tree.query(xyz3, k=20)
        idxnew = np.unique(idxnew).T
    
        # Cell radius
        dx = 0.5 * Vol[idx][underflow_mask]**(1/3)
        
        # Get the Grads    
        f_inter_input = np.array([ X[idxnew], Y[idxnew], Z[idxnew] ]).T
    
        gradx_p = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                            xi = np.array([ ray_x+dx, ray_y, ray_z]).T )
        gradx_m = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                            xi = np.array([ ray_x-dx, ray_y, ray_z]).T )
        gradx = (gradx_p - gradx_m)/ (2*dx)
        gradx = np.nan_to_num(gradx, nan =  0)
        del gradx_p, gradx_m
    
        grady_p = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                            xi = np.array([ ray_x, ray_y+dx, ray_z]).T )
        grady_m = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                            xi = np.array([ ray_x, ray_y-dx, ray_z]).T )
        grady = (grady_p - grady_m)/ (2*dx)
        grady = np.nan_to_num(grady, nan =  0)
        del grady_p, grady_m
    
        gradz_p = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                            xi = np.array([ ray_x, ray_y, ray_z+dx]).T )
        gradz_m = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                            xi = np.array([ ray_x, ray_y, ray_z-dx]).T )
        gradz_m = np.nan_to_num(gradz_m, nan =  0)
        gradz = (gradz_p - gradz_m)/ (2*dx)
        del gradz_p, gradz_m
    
        grad = np.sqrt(gradx**2 + grady**2 + gradz**2)
        gradr = (mu_x * gradx) + (mu_y*grady) + (mu_z*gradz)
        del gradx, grady, gradz
        gc.collect()
        
        R_lamda = grad / ( c.Rsol_to_cm * sigma_rossland_eval* rad_den)
        R_lamda[R_lamda < 1e-10] = 1e-10
        
        fld_factor = 3 * (1/np.tanh(R_lamda) - 1/R_lamda) / R_lamda 
        smoothed_flux = -uniform_filter1d(r.T**2 * fld_factor * gradr / sigma_rossland_eval, 7) 
        photosphere = np.where( ((smoothed_flux>0) & (los<2/3) ))[0][0]

        
        Lphoto2 = 4*np.pi*c.c*smoothed_flux[photosphere] * c.Msol_to_g / (c.t**2)
        if Lphoto2 < 0:
            Lphoto2 = 1e100 # it means that it will always pick max_length for the negatives, maybe this is what we are getting wrong
        max_length = 4*np.pi*c.c*rad_den[photosphere]*r[photosphere]**2 * c.Msol_to_g * c.Rsol_to_cm / (c.t**2)
        reds[obs] = np.min( [Lphoto2, max_length])
        print(reds[obs])
        
    red =  np.mean(reds) # DO NOT ADD A 4pi HERE
    if save and alice: # Save red
        pre_saving = '/home/kilmetisk/data1/TDE/tde_comparison/data/'
        if single:
            filepath =  f'{pre_saving}red/red_walljumper2{m}.csv'
            data = [fix, day, red]
            with open(filepath, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(data)
            file.close()

        
        
