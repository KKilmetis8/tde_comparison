#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23

@author: konstantinos
"""
import sys
import gc
import time
sys.path.append('/Users/paolamartire/tde_comparison')
import warnings
warnings.filterwarnings('ignore')
import csv


# The goal is to replicate Elad's script. No nice code, no nothing. A shit ton
# of comments though. 
import numpy as np
import healpy as hp
import scipy.integrate as sci
from scipy.interpolate import griddata
import matlab.engine
from sklearn.neighbors import KDTree

from src.Utilities.isalice import isalice
alice, plot = isalice()
import src.Utilities.prelude as c
from scipy.ndimage import uniform_filter1d # does moving mean without fucking the shape up
import src.Utilities.selectors as s
from src.Utilities.parser import parse

# Okay, import the constants. Do not be absolutely terrible'''

#%% Choose parameters -----------------------------------------------------------------
save = True
if alice:
    pre = '/home/kilmetisk/data1/TDE/'
    args = parse()
    sim = args.name
    mstar = args.mass
    rstar = args.radius
    Mbh = args.blackhole
    fixes = np.arange(args.first, args.last + 1)
    single = args.single
    if single:
        fixes = [args.only]
    else:
        Lphoto_all = np.zeros(len(fixes))
    opac_kind = 'LTE'
    m = 'AEK'
    check = 'MONO AEK'
else:
    m = 5
    pre = f'{m}/'
    fixes = [236]
    opac_kind = 'LTE'
    mstar = 0.5
    rstar = 0.47


#%% Opacities -----------------------------------------------------------------
# Freq range
f_min = c.Kb * 1e3 / c.h
f_max = c.Kb * 3e13 / c.h
f_num = 1_000
frequencies = np.logspace(np.log10(f_min), np.log10(f_max), f_num)

# Opacity Input
opac_path = f'src/Opacity/{opac_kind}_data/'
T_cool = np.loadtxt(f'{opac_path}/T.txt')
Rho_cool = np.loadtxt(f'{opac_path}/rho.txt')
plank = np.loadtxt(f'{opac_path}/planck.txt')
rossland = np.loadtxt(f'{opac_path}/ross.txt')

# Fill value none extrapolates
def linearpad(D0,z0):
    factor = 100
    dz = z0[-1] - z0[-2]
    # print(np.shape(D0))
    dD = D0[:,-1] - D0[:,-2]
    
    z = [zi for zi in z0]
    z.append(z[-1] + factor*dz)
    z = np.array(z)
    #D = [di for di in D0]

    to_stack = np.add(D0[:,-1], factor*dD)
    to_stack = np.reshape(to_stack, (len(to_stack),1) )
    D = np.hstack((D0, to_stack))
    #D.append(to_stack)
    return np.array(D), z

def pad_interp(x,y,V):
    Vn, xn = linearpad(V, x)
    Vn, xn = linearpad(np.fliplr(Vn), np.flip(xn))
    Vn = Vn.T
    Vn, yn = linearpad(Vn, y)
    Vn, yn = linearpad(np.fliplr(Vn), np.flip(yn))
    Vn = Vn.T
    return xn, yn, Vn

T_cool2, Rho_cool2, rossland2 = pad_interp(T_cool, Rho_cool, rossland.T)
_, _, plank2 = pad_interp(T_cool, Rho_cool, plank.T)

# MATLAB GOES WHRRRR, thanks Cindy.
eng = matlab.engine.start_matlab()

days = []
for idx_s, snap in enumerate(fixes):
    #%% Load data -----------------------------------------------------------------
    if alice:
            X = np.load(f'{pre}{sim}/snap_{snap}/CMx_{snap}.npy')
            Y = np.load(f'{pre}{sim}/snap_{snap}/CMy_{snap}.npy')
            Z = np.load(f'{pre}{sim}/snap_{snap}/CMz_{snap}.npy')
            T = np.load(f'{pre}{sim}/snap_{snap}/T_{snap}.npy')
            Den = np.load(f'{pre}{sim}/snap_{snap}/Den_{snap}.npy')
            Rad = np.load(f'{pre}{sim}/snap_{snap}/Rad_{snap}.npy')
            Vol = np.load(f'{pre}{sim}/snap_{snap}/Vol_{snap}.npy')
            day = np.loadtxt(f'{pre}{sim}/snap_{snap}/tbytfb_{snap}.txt')
            box = np.load(f'{pre}{sim}/snap_{snap}/box_{snap}.npy')
            days.append(day)
    else:
        X = np.load(f'{pre}{snap}/CMx_{snap}.npy')
        Y = np.load(f'{pre}{snap}/CMy_{snap}.npy')
        Z = np.load(f'{pre}{snap}/CMz_{snap}.npy')
        # VX = np.load(f'{pre}{snap}/Vx_{snap}.npy')
        # VY = np.load(f'{pre}{snap}/Vy_{snap}.npy')
        # VZ = np.load(f'{pre}{snap}/Vz_{snap}.npy')
        T = np.load(f'{pre}{snap}/T_{snap}.npy')
        Den = np.load(f'{pre}{snap}/Den_{snap}.npy')
        Rad = np.load(f'{pre}{snap}/Rad_{snap}.npy')
        # IE = np.load(f'{pre}{snap}/IE_{snap}.npy')
        Vol = np.load(f'{pre}{snap}/Vol_{snap}.npy')
        #day = np.loadtxt(f'{pre}{sim}/snap_{snap}/tbytfb_{snap}.txt')
        box = np.load(f'{pre}{snap}/box_{snap}.npy')
        #days.append(day)
        #del day
    denmask = Den > 1e-19
    X = X[denmask]
    Y = Y[denmask]
    Z = Z[denmask]
    T = T[denmask]
    Den = Den[denmask]

    Rad = Rad[denmask]
    Vol = Vol[denmask]
    Rad_den = np.multiply(Rad,Den)
    del Rad            
    R = np.sqrt(X**2 + Y**2 + Z**2)
    #%% Cross dot -----------------------------------------------------------------
    observers_xyz = hp.pix2vec(c.NSIDE, range(c.NPIX))
    # Line 17, * is matrix multiplication, ' is .T
    observers_xyz = np.array(observers_xyz).T
    cross_dot = np.matmul(observers_xyz,  observers_xyz.T)
    cross_dot[cross_dot<0] = 0
    cross_dot *= 4/c.NPIX

    #%% Tree ----------------------------------------------------------------------
    #from scipy.spatial import KDTree
    xyz = np.array([X, Y, Z]).T
    N_ray = 5_000

    # Flux?
    F_photo = np.zeros((c.NPIX, f_num))
    F_photo_temp = np.zeros((c.NPIX, f_num))

    # Lines 99-128 use some files we don't have, I think we only need
    # these for blue. Ignore for now 

    # Dynamic Box -----------------------------------------------------------------
    reds = np.zeros(c.NPIX)
    time_start = 0
    for i in range(c.NPIX):
        # Progress 
        time_end = time.time()
        print(f'Snap: {snap}, Obs: {i}', flush=False)
        print(f'Time for prev. Obs: {(time_end - time_start)/60} min', flush = False)
        time_start = time.time()
        sys.stdout.flush()

        # if i % 10 == 0:
        #     print('Eladython Ray no:', i)
        # print(i)
        mu_x = observers_xyz[i][0]
        mu_y = observers_xyz[i][1]
        mu_z = observers_xyz[i][2]

        # Box is for dynamic ray making
        if mu_x < 0:
            rmax = box[0] / mu_x
            # print('x-', rmax)
        else:
            rmax = box[3] / mu_x
            # print('x+', rmax)
        if mu_y < 0:
            rmax = min(rmax, box[1] / mu_y)
            # print('y-', rmax)
        else:
            rmax = min(rmax, box[4] / mu_y)
            # print('y+', rmax)

        if mu_z < 0:
            rmax = min(rmax, box[2] / mu_z)
            # print('z-', rmax)
        else:
            rmax = min(rmax, box[5] / mu_z)
            # print('z+', rmax)

        # This naming is so bad
        r = np.logspace( -0.25, np.log10(rmax), N_ray)
        alpha = (r[1] - r[0]) / (0.5 * ( r[0] + r[1]))
        dr = alpha * r

        x = r*mu_x
        y = r*mu_y
        z = r*mu_z
        xyz2 = np.array([x, y, z]).T
        del x, y, z
        tree = KDTree(xyz, leaf_size=50)
        _, idx = tree.query(xyz2, k=1)
        idx = [ int(idx[i][0]) for i in range(len(idx))] # no -1 because we start from 0
        d = Den[idx] * c.den_converter
        t = T[idx]

        # Interpolate ----------------------------------------------------------
        sigma_rossland = eng.interp2(T_cool2,Rho_cool2,rossland2,np.log(t),np.log(d),'linear',0)
        sigma_rossland = [sigma_rossland[0][i] for i in range(N_ray)]
        sigma_rossland_eval = np.exp(sigma_rossland) 

        sigma_plank = eng.interp2(T_cool2,Rho_cool2,plank2,np.log(t),np.log(d),'linear',0)
        sigma_plank = [sigma_plank[0][i] for i in range(N_ray)]
        sigma_plank_eval = np.exp(sigma_plank)
        del sigma_rossland, sigma_plank 
        gc.collect()

        # Optical Depth ---------------------------------------------------------------
        # Okay, line 232, this is the hard one.
        r_fuT = np.flipud(r.T)
        kappa_rossland = np.flipud(sigma_rossland_eval) 
        los = - np.flipud(sci.cumulative_trapezoid(kappa_rossland, r_fuT, initial = 0)) * c.Rsol_to_cm # dont know what it do but this is the conversion
        k_effective = np.sqrt(3 * np.flipud(sigma_plank_eval) * np.flipud(sigma_rossland_eval)) 
        los_effective = - np.flipud(sci.cumulative_trapezoid(k_effective, r_fuT, initial = 0)) * c.Rsol_to_cm

        # Red -----------------------------------------------------------------------
        # Get 20 unique, nearest neighbors
        xyz3 = np.array([X[idx], Y[idx], Z[idx]]).T
        xyz3 = np.array([X[idx], Y[idx], Z[idx]]).T
        _, idxnew = tree.query(xyz3, k=20)
        idxnew = np.unique(idxnew).T
        dx = 0.5 * Vol[idx]**(1/3) # Cell radius

        # Get the Grads
        # sphere and get the gradient on them. Is it neccecery to re-interpolate?
        # scattered interpolant returns a function
        # griddata DEMANDS that you pass it the values you want to eval at
        f_inter_input = np.array([ X[idxnew], Y[idxnew], Z[idxnew] ]).T

        gradx_p = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                            xi = np.array([ X[idx]+dx, Y[idx], Z[idx]]).T )
        gradx_m = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                            xi = np.array([ X[idx]-dx, Y[idx], Z[idx]]).T )
        gradx = (gradx_p - gradx_m)/ (2*dx)
        del gradx_p, gradx_m

        gradx = np.nan_to_num(gradx, nan =  0)
        grady_p = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                            xi = np.array([ X[idx], Y[idx]+dx, Z[idx]]).T )
        grady_m = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                            xi = np.array([ X[idx], Y[idx]-dx, Z[idx]]).T )
        grady = (grady_p - grady_m)/ (2*dx)
        del grady_p, grady_m

        grady = np.nan_to_num(grady, nan =  0)

        gradz_p = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                            xi = np.array([ X[idx], Y[idx], Z[idx]+dx]).T )
        gradz_m = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                            xi = np.array([ X[idx], Y[idx], Z[idx]-dx]).T )
        # some nans here
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
        smoothed_flux = -uniform_filter1d(r.T**2 * fld_factor * gradr / sigma_rossland_eval, 7) # i have remov
        # Spectra --------------------------------------------------------------
        F_photo_temp = np.zeros((c.NPIX, f_num))
        try:
            b = np.where( ((smoothed_flux>0) & (los<2/3) ))[0][0] 
        except IndexError:
            b = 3117 # elad_b = 3117
        Lphoto2 = 4*np.pi*c.c*smoothed_flux[b] * c.Msol_to_g / (c.t**2)
        EEr = Rad_den[idx]
        if Lphoto2 < 0:
            Lphoto2 = 1e100 # it means that it will always pick max_length for the negatives
        max_length = 4*np.pi*c.c*EEr[b]*r[b]**2 * c.Msol_to_g * c.Rsol_to_cm / (c.t**2)
        Lphoto = np.min( [Lphoto2, max_length])
        reds[i] = Lphoto
        del smoothed_flux, R_lamda, fld_factor, EEr, los,
        gc.collect()
if single:
    Lphoto_this = np.mean(reds)
else:
    Lphoto_all[idx_s] = np.mean(reds) # save red
    # Lphoto = Lphoto2
if save:
    if alice:
        pre_saving = f'/home/kilmetisk/data1/TDE/tde_comparison/data/'
        if single:
            filepath =  f'{pre_saving}red/{sim}/eladred_d19.csv'
            data = [snap, day, Lphoto_this]
            with open(filepath, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(data)
            file.close()
        else:
            np.savetxt(f'{pre_saving}red/{sim}_eladred_{args.first}to{args.last}.txt', Lphoto_all)
            np.savetxt(f'{pre_saving}red/{sim}_eladreddays_{args.first}to{args.last}.txt', days)
        print('saved red and days')
eng.exit()