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
eng = matlab.engine.start_matlab()
from sklearn.neighbors import KDTree
from scipy.ndimage import uniform_filter1d

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
def masker(mask, list_of_quantities):
    new_list = []
    for quantity in list_of_quantities:
        new_list.append(quantity[mask])
    return (*new_list,)

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
    X, Y, Z, Den, T, Rad, Vol, box, day = alice_loader(sim, fix, 'thermodynamics')
else:
    m = 5
    pre = f'{m}/'
    fix = 288
    sim = ''
    mstar = 0.5
    rstar = 0.47
    X, Y, Z, Den, T, Rad, Vol, box, day = local_loader(m, fix, 'thermodynamics')  

Rad_den = np.multiply(Rad, Den)
R = np.sqrt(X**2 + Y**2 + Z**2)

# Observers initialization ---
observers = np.arange(0,c.NPIX)
observers_xyz = hp.pix2vec(c.NSIDE, range(c.NPIX))
observers_xyz = np.array([observers_xyz]).T[:,:,0]
cross_dot = np.matmul(observers_xyz,  observers_xyz.T )
cross_dot[cross_dot<0] = 0
cross_dot *= 4/192

# Freq range
reds = np.zeros(c.NPIX)
N_ray = 5_000
frequencies = c.freqs

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


### Do it --- --- ---
xyz = np.array([X, Y, Z]).T

# Flux
F_photo = np.zeros((c.NPIX, len(frequencies)))
F_photo_temp = np.zeros((c.NPIX, len(frequencies)))
photosphere = []
colorsphere = []
time_start = 0
reds = np.zeros(c.NPIX)
observers = np.arange(0,c.NPIX)

for i, obs in enumerate(observers):
    # Progress 
    time_end = time.time()
    print(f'Snap: {fix}, Obs: {i}', 
          flush=False)
    print(f'Time for prev. Obs: {(time_end - time_start)/60} min', 
          flush = False)
    time_start = time.time()
    sys.stdout.flush()
    

    # Make the ray
    rmax = boxer(obs, observers_xyz, box)
    r = np.logspace( -0.25, np.log10(rmax), N_ray)
    mu_x = observers_xyz[obs][0]
    mu_y = observers_xyz[obs][1]
    mu_z = observers_xyz[obs][2]
    x = r*mu_x
    y = r*mu_y
    z = r*mu_z
    xyz2 = np.array([x, y, z]).T
    tree = KDTree(xyz, leaf_size=25)
    _, idx = tree.query(xyz2, k=1)
    idx = [ int(idx[i][0]) for i in range(len(idx))] # no -1 because we start from 0
    del x, y, z

    d = Den[idx] * c.den_converter
    t = T[idx]
    ray_x = X[idx]
    ray_y = Y[idx]
    ray_z = Z[idx]
    rad_den = Rad_den[idx]

    # Interpolate ---
    sigma_plank = eng.interp2(T_cool2,Rho_cool2,plank2.T, 
                              np.log(t),np.log(d),'linear',0)
    sigma_plank = np.array(sigma_plank)[0]
    underflow_mask_plank = sigma_plank != 0.0

    sigma_scattering = eng.interp2(T_cool2,Rho_cool2,scattering2.T, 
                              np.log(t),np.log(d),'linear',0)
    sigma_scattering = np.array(sigma_scattering)[0]
    underflow_mask_scattering = sigma_scattering != 0.0
    underflow_mask = underflow_mask_scattering * underflow_mask_plank
    
    d, t, r, sigma_plank, sigma_scattering, ray_x, ray_y, ray_z, rad_den = masker(underflow_mask, 
    [d, t, r, sigma_plank, sigma_scattering, ray_x, ray_y, ray_z, rad_den])
    sigma_plank_eval = np.exp(sigma_plank)
    sigma_scattering_eval = np.exp(sigma_scattering)
    sigma_rossland_eval = sigma_plank_eval + sigma_scattering_eval

    # Optical Depth ---.    
    r_fuT = np.flipud(r.T)
    
    # Try out elads scattering + rosseland
    kappa_rossland = np.flipud(sigma_scattering_eval) + np.flipud(sigma_plank_eval)
    los = - np.flipud(sci.cumulative_trapezoid(kappa_rossland, 
                                               r_fuT, initial = 0)) * c.Rsol_to_cm # dont know what it do but this is the conversion
    
    kappa_plank = np.flipud(sigma_plank_eval) 
    los_abs = - np.flipud(sci.cumulative_trapezoid(kappa_plank,
                                                   r_fuT, initial = 0)) * c.Rsol_to_cm
    k_effective = np.sqrt(3 * np.flipud(sigma_plank_eval) * kappa_rossland) 
    los_effective = - np.flipud(sci.cumulative_trapezoid(k_effective, 
                                                         r_fuT, initial = 0)) * c.Rsol_to_cm
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
    
    R_lamda = grad / ( c.Rsol_to_cm * sigma_rossland_eval * rad_den)
    R_lamda[R_lamda < 1e-10] = 1e-10
    
    fld_factor = (1/np.tanh(R_lamda) - 1/R_lamda) / R_lamda 
    smoothed_flux = -uniform_filter1d(r.T**2 * fld_factor * gradr / sigma_rossland_eval, 7) 
    photo_idx = np.where( ((smoothed_flux>0) & (los<2/3) ))[0][0]
    
    Lphoto2 = 4*np.pi*c.c*smoothed_flux[photo_idx] * c.Msol_to_g / (c.t**2)
    if Lphoto2 < 0:
        Lphoto2 = 1e100 # it means that it will always pick max_length for the negatives, maybe this is what we are getting wrong
    max_length = 4*np.pi*c.c*rad_den[photo_idx]*r[photo_idx]**2 * c.Msol_to_g * c.Rsol_to_cm / (c.t**2)
    reds[i] = np.min( [Lphoto2, max_length])
        
    # Spectra
    los_effective[los_effective>30] = 30
    color_idx = np.argmin(np.abs(los_effective-5))
    
    photosphere.append((ray_x[photo_idx], ray_y[photo_idx], ray_z[photo_idx]))
    colorsphere.append((ray_x[color_idx], ray_y[color_idx], ray_z[color_idx])) 
    
    # Spectra ---
    for k in range(color_idx, len(r)):
        # tlim = 1e20
        # if t[k] < tlim:
        dr = r[k]-r[k-1]
        Vcell =  r[k]**2 * dr # there should be a (4 * np.pi / 192)*, but doesn't matter because we normalize
        wien = np.exp(c.h * frequencies / (c.kb * t[k])) - 1
        black_body = frequencies**3 / (c.c**2 * wien)
    # print(sigma_plank_eval[k] * Vcell * np.exp(-los_effective[k]) * black_body )
        F_photo_temp[obs,:] += sigma_plank_eval[k] * Vcell * np.exp(-los_effective[k]) * black_body
    
    norm = reds[i] / np.trapz(F_photo_temp[i,:], frequencies)
    F_photo_temp[i,:] *= norm
    F_photo[i,:] = np.dot(cross_dot[i,:], F_photo_temp)    
eng.exit()

### Bolometric ---
red = np.mean(reds) # DO NOT put a 4pi here

#%% Saving ---
if save and alice: # Save red
        pre_saving = '/home/kilmetisk/data1/TDE/tde_comparison/data/'
        if single:
            filepath =  f'{pre_saving}red/red_fromsum{m}.csv'
            data = [fix, day, red]
            with open(filepath, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(data)
            file.close()
            
        # Save spectrum
        np.savetxt(f'{pre_saving}blue/{sim}/freqs.txt', frequencies)
        np.savetxt(f'{pre_saving}blue/{sim}/walljump_s3_{m}spectra{fix}.txt', F_photo)
        
        # Save photocolor
        filepath =  f'{pre_saving}photosphere/walljump_s3_photocolor{m}.csv'
        data = [fix, day, 
                np.mean( np.linalg.norm(photosphere, axis = 1)), 
                np.mean( np.linalg.norm(colorsphere, axis = 1)), 
                c.NPIX]
        data.append( np.linalg.norm(photosphere, axis = 1))
        data.append( np.linalg.norm(colorsphere, axis = 1))
        
        with open(filepath, 'a', newline='') as file:
            file.write('# snap, time [tfb], photo [Rsol], color [Rsol], NPIX, NPIX cols with photo for each observer, NPIX cols with color for each observer \n')
            writer = csv.writer(file)
            writer.writerow(data)
        file.close()
if save and not alice:
        # Save photocolor
        pre_saving = 'data/bluepaper/'
        np.savetxt(f'{pre_saving}localbig_{m}spectra{fix}.txt', F_photo)

        filepath =  f'{pre_saving}/localbig_{m}photocolor{fix}.csv'
        data = [fix, day, 
                np.mean( np.linalg.norm(photosphere, axis = 1)), 
                np.mean( np.linalg.norm(colorsphere, axis = 1)), 
                c.NPIX]
        data.append( np.linalg.norm(photosphere, axis = 1))
        data.append( np.linalg.norm(colorsphere, axis = 1))
        
        with open(filepath, 'a', newline='') as file:
            # file.write('# snap, time [tfb], photo [Rsol], color [Rsol], NPIX, NPIX cols with photo for each observer, NPIX cols with color for each observer \n')
            writer = csv.writer(file)
            writer.writerow(data)
        file.close()
