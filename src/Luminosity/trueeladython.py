#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 18:52:21 2024

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

from src.Opacity.LTE_loader import T_opac_ex, Rho_opac_ex, rossland_ex, plank_ex
import src.Utilities.prelude as c
from src.Utilities.parser import parse
from src.Utilities.isalice import isalice
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
    m = 4
    pre = f'{m}/'
    fix = 272
    sim = ''
    mstar = 0.5
    rstar = 0.47

X = np.load(f'{pre}{fix}/CMx_{fix}.npy')
Y = np.load(f'{pre}{fix}/CMy_{fix}.npy')
Z = np.load(f'{pre}{fix}/CMz_{fix}.npy')
T = np.load(f'{pre}{fix}/T_{fix}.npy')
Den = np.load(f'{pre}{fix}/Den_{fix}.npy')
Rad = np.load(f'{pre}{fix}/Rad_{fix}.npy')
Vol = np.load(f'{pre}{fix}/Vol_{fix}.npy')
box = np.load(f'{pre}{fix}/box_{fix}.npy')
day = np.loadtxt(f'{pre}{fix}/tbytfb_{fix}.txt')

Rad_den = np.multiply(Rad,Den)
R = np.sqrt(X**2 + Y**2 + Z**2)

# Cross dot ---
observers_xyz = hp.pix2vec(c.NSIDE, range(192))
observers_xyz = np.array([observers_xyz]).T
# Line 17, * is matrix multiplication, ' is .T
cross_dot = np.matmul(observers_xyz,  observers_xyz.T )
cross_dot[cross_dot<0] = 0
cross_dot *= 4/192

# <dubious code>  -----
# Correction!
# npix = hp.nside2npix(c.NSIDE)
# theta, phi = hp.pix2ang(c.NSIDE, range(npix))
# outx = np.sin(theta) * np.cos(phi)
# outy = np.sin(theta) * np.sin(phi)
# outz = np.cos(theta)
# outX = np.array([outx, outy, outz]).T
# cross_dot2 = np.dot(outX,  outX.T )
# cross_dot2[cross_dot2<0] = 0
# cross_dot2 *= 4/192
# cross_dot = cross_dot2
# <\dubious code>  -----

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

### Do it --- --- ---
eng = matlab.engine.start_matlab()
xyz = np.array([X, Y, Z]).T

# Flux
F_photo = np.zeros((c.NPIX, f_num))
F_photo_temp = np.zeros((c.NPIX, f_num))
photosphere = []
colorsphere = []
time_start = 0
reds = np.zeros(c.NPIX)

# Iterate over observers
for i in range(c.NPIX):
    # Progress 
    time_end = time.time()
    print(f'Snap: {fix}, Obs: {i}', 
          flush=False)
    print(f'Time for prev. Obs: {(time_end - time_start)/60} min', 
          flush = False)
    time_start = time.time()
    sys.stdout.flush()
    
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
    r = np.logspace( -0.25, np.log10(rmax), N_ray)
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
    sigma_rossland_eval = np.exp(sigma_rossland) 
    
    sigma_plank = eng.interp2(T_cool2,Rho_cool2,plank2.T, 
                              np.log(t),np.log(d),'linear',0)
    sigma_plank = np.array(sigma_plank)[0]
    sigma_plank_eval = np.exp(sigma_plank)
    del sigma_rossland, sigma_plank 
    gc.collect()
    
    # Optical Depth ---.    
    r_fuT = np.flipud(r.T)
    kappa_rossland = np.flipud(sigma_rossland_eval) 
    los = - np.flipud(sci.cumulative_trapezoid(kappa_rossland, r_fuT, initial = 0)) * c.Rsol_to_cm # dont know what it do but this is the conversion
    
    kappa_plank = np.flipud(sigma_plank_eval) 
    los_abs = - np.flipud(sci.cumulative_trapezoid(kappa_plank, r_fuT, initial = 0)) * c.Rsol_to_cm
    k_effective = np.sqrt(3 * np.flipud(sigma_plank_eval) * np.flipud(sigma_rossland_eval)) 
    los_effective = - np.flipud(sci.cumulative_trapezoid(k_effective, r_fuT, initial = 0)) * c.Rsol_to_cm
    
    # Red ---
    # Get 20 unique, nearest neighbors
    xyz3 = np.array([X[idx], Y[idx], Z[idx]]).T
    xyz3 = np.array([X[idx], Y[idx], Z[idx]]).T
    _, idxnew = tree.query(xyz3, k=20)
    idxnew = np.unique(idxnew).T

    # Cell radius
    dx = 0.5 * Vol[idx]**(1/3)
    
    # Get the Grads    
    f_inter_input = np.array([ X[idxnew], Y[idxnew], Z[idxnew] ]).T

    gradx_p = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                        xi = np.array([ X[idx]+dx, Y[idx], Z[idx]]).T )
    gradx_m = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                        xi = np.array([ X[idx]-dx, Y[idx], Z[idx]]).T )
    gradx = (gradx_p - gradx_m)/ (2*dx)
    gradx = np.nan_to_num(gradx, nan =  0)
    del gradx_p, gradx_m

    grady_p = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                        xi = np.array([ X[idx], Y[idx]+dx, Z[idx]]).T )
    grady_m = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                        xi = np.array([ X[idx], Y[idx]-dx, Z[idx]]).T )
    grady = (grady_p - grady_m)/ (2*dx)
    grady = np.nan_to_num(grady, nan =  0)
    del grady_p, grady_m

    gradz_p = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                        xi = np.array([ X[idx], Y[idx], Z[idx]+dx]).T )
    gradz_m = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                        xi = np.array([ X[idx], Y[idx], Z[idx]-dx]).T )
    gradz_m = np.nan_to_num(gradz_m, nan =  0)
    gradz = (gradz_p - gradz_m)/ (2*dx)
    del gradz_p, gradz_m

    grad = np.sqrt(gradx**2 + grady**2 + gradz**2)
    gradr = (mu_x * gradx) + (mu_y*grady) + (mu_z*gradz)
    del gradx, grady, gradz
    gc.collect()
    
    R_lamda = grad / ( c.Rsol_to_cm * sigma_rossland_eval* Rad_den[idx])
    R_lamda[R_lamda < 1e-10] = 1e-10
    fld_factor = 3 * (1/np.tanh(R_lamda) - 1/R_lamda) / R_lamda 
    smoothed_flux = -uniform_filter1d(r.T**2 * fld_factor * gradr / sigma_rossland_eval, 7) 
    
    # Spectra
    try:
        b = np.where( ((smoothed_flux>0) & (los<2/3) ))[0][0]
    except:
        print(f'\n {i}, elad b')
        b = 3117
        
    los_effective[los_effective>30] = 30
    b2 = np.argmin(np.abs(los_effective-5))
    
    photosphere.append((ray_x[b], ray_y[b], ray_z[b]))
    colorsphere.append((ray_x[b2], ray_y[b2], ray_z[b2])) 
    
    Lphoto2 = 4*np.pi*c.c*smoothed_flux[b] * c.Msol_to_g / (c.t**2)
    EEr = Rad_den[idx]
    if Lphoto2 < 0:
        Lphoto2 = 1e100 # it means that it will always pick max_length for the negatives, maybe this is what we are getting wrong
    max_length = 4*np.pi*c.c*EEr[b]*r[b]**2 * c.Msol_to_g * c.Rsol_to_cm / (c.t**2)
    reds[i] = np.min( [Lphoto2, max_length])
    del smoothed_flux, R_lamda, fld_factor, EEr, los,
    gc.collect()
    # Spectra ---
    los_effective[los_effective>30] = 30
    b2 = np.argmin(np.abs(los_effective-5))

    for k in range(b2, len(r)):
        dr = r[k]-r[k-1]
        Vcell =  r[k]**2 * dr # there should be a (4 * np.pi / 192)*, but doesn't matter because we normalize
        wien = np.exp(c.h * frequencies / (c.kb * t[k])) - 1
        black_body = frequencies**3 / (c.c**2 * wien)
        F_photo_temp[i,:] += sigma_plank_eval[k] * Vcell * np.exp(-los_effective[k]) * black_body

    norm = reds[i] / np.trapz(F_photo_temp[i,:], frequencies)
    F_photo_temp[i,:] *= norm
    F_photo[i,:] = np.dot(cross_dot[i,:], F_photo_temp)      
eng.exit()

### Bolometric ---
red = 4 * np.pi * np.mean(reds)

### Saving ---
if save and alice: # Save red
        pre_saving = '/home/kilmetisk/data1/TDE/tde_comparison/data/'
        if single:
            filepath =  f'{pre_saving}red/red_richex{m}.csv'
            data = [fix, day, red]
            with open(filepath, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(data)
            file.close()
            
        # Save spectrum
        np.savetxt(f'{pre_saving}blue/{sim}/freqs.txt', frequencies)
        np.savetxt(f'{pre_saving}blue/{sim}/richex_{m}spectra{fix}.txt', F_photo)
        
        # Save photocolor
        filepath =  f'{pre_saving}photosphere/richex_photocolor{m}.csv'
        data = [fix, day, np.mean(photosphere), np.mean(colorsphere), c.NPIX]
        [ data.append(photosphere[i]) for i in range(c.NPIX)]
        [ data.append(colorsphere[i]) for i in range(c.NPIX)]
        
        with open(filepath, 'a', newline='') as file:
            file.write('# snap, time [tfb], photo [Rsol], color [Rsol], NPIX, NPIX cols with photo for each observer, NPIX cols with color for each observer \n')
            writer = csv.writer(file)
            writer.writerow(data)
        file.close()
