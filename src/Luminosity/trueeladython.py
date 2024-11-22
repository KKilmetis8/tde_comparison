#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:53:18 2024

@author: konstantinos
"""

import sys
sys.path.append('/Users/paolamartire/tde_comparison')
import warnings
warnings.filterwarnings('ignore')
# The goal is to replicate Elad's script. No nice code, no nothing. A shit ton
# of comments though. 
import h5py
import gc
import numpy as np

import matplotlib.pyplot as plt
import healpy as hp
import scipy.integrate as sci
from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator
import matlab.engine
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors, KDTree
from datetime import datetime
from scipy.ndimage import uniform_filter1d # does moving mean without fucking the shape up

from src.Utilities.isalice import isalice
alice, plot = isalice()
import src.Utilities.prelude as c
import src.Utilities.selectors as s
from src.Utilities.parser import parse

# Okay, import the constants. Do not be absolutely terrible'''
    # n_start = snap_no_start
    # n_end = snap_no_end
pstart = datetime.now()

#%% Choose parameters -----------------------------------------------------------------
save = False
if alice:
    pre = '/home/s3745597/data1/TDE/'
    args = parse()
    sim = args.name
    mstar = args.mass
    rstar = args.radius
    Mbh = args.blackhole
    # fixes = np.arange(args.first, args.last + 1)
    opac_kind = 'LTE'
    m = 'AEK'
    check = 'MONO AEK'

    if Mbh == 10_000:
        if 'HiRes' in sim:
            fixes = [210]
            print('BH 4 HiRes')

        else:
            fixes = [272] # [164, 237, 313]
            print('BH 4')
    elif Mbh == 100_000:
        fixes = [302] # [208, 268,]# 365]
        print('BH 5')
    else:
        fixes = [351]
        print('BH 6')
else:
    m = 5
    pre = f'{m}/'
    fixes = [302] #[301, 302, 303]
    opac_kind = 'LTE'
    mstar = 0.5
    rstar = 0.47

#%% Opacities -----------------------------------------------------------------
# Freq range
f_min = c.Kb * 1e3 / c.h
f_max = c.Kb * 3e10 / c.h
f_num = 1_000
frequencies = np.logspace(np.log10(f_min), np.log10(f_max), f_num)

# Opacity Input
opac_path = f'src/Opacity/{opac_kind}_data/'
T_cool = np.loadtxt(f'{opac_path}/T.txt')
Rho_cool = np.loadtxt(f'{opac_path}/rho.txt')
plank = np.loadtxt(f'{opac_path}/planck.txt')
rossland = np.loadtxt(f'{opac_path}/ross.txt')
scatter = np.loadtxt(f'{opac_path}/scatter.txt')

tau_eff = []
tau_ross = []
tau_abs = []
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
_, _, scatter2 = pad_interp(T_cool, Rho_cool, scatter.T)


# MATLAB GOES WHRRRR, thanks Cindy.
eng = matlab.engine.start_matlab()

N_ray = 5_000
Lphoto_all = np.zeros(len(fixes))
Lphoto_this = np.zeros((c.NPIX))
photosphere = np.zeros((c.NPIX, len(fixes)))
photo_d = np.zeros((c.NPIX, len(fixes)))
photo_t = np.zeros((c.NPIX, len(fixes)))

therm_radius = np.zeros((c.NPIX, len(fixes)))
therm_d = np.zeros((c.NPIX, len(fixes)))
therm_t = np.zeros((c.NPIX, len(fixes)))
days = []
rays = np.arange(0,c.NPIX) # []
denplot = []
tplot = []
radii = []
for idx_s, snap in enumerate(fixes):
    print('Snapshot: ', snap)
    temperatures = np.zeros((N_ray, c.NPIX))
    # Load data -----------------------------------------------------------------
    if alice:
        X = np.load(f'{pre}{sim}/snap_{snap}/CMx_{snap}.npy')
        Y = np.load(f'{pre}{sim}/snap_{snap}/CMy_{snap}.npy')
        Z = np.load(f'{pre}{sim}/snap_{snap}/CMz_{snap}.npy')
        # VX = np.load(f'{pre}{sim}/snap_{snap}/Vx_{snap}.npy')
        # VY = np.load(f'{pre}{sim}/snap_{snap}/Vy_{snap}.npy')
        # VZ = np.load(f'{pre}{sim}/snap_{snap}/Vz_{snap}.npy')
        T = np.load(f'{pre}{sim}/snap_{snap}/T_{snap}.npy')
        Den = np.load(f'{pre}{sim}/snap_{snap}/Den_{snap}.npy')
        Rad = np.load(f'{pre}{sim}/snap_{snap}/Rad_{snap}.npy')
        # IE = np.load(f'{pre}{sim}/snap_{snap}/IE_{snap}.npy')
        Vol = np.load(f'{pre}{sim}/snap_{snap}/Vol_{snap}.npy')
        day = np.loadtxt(f'{pre}{sim}/snap_{snap}/tbytfb_{snap}.txt')
        box = np.load(f'{pre}{sim}/snap_{snap}/box_{snap}.npy')
        days.append(day)
        del day
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
        day = np.loadtxt(f'{pre}{snap}/tbytfb_{snap}.txt')
        box = np.load(f'{pre}{snap}/box_{snap}.npy')
        #days.append(day)
        #del day
    denmask = Den > 1e-21
    X = X[denmask]
    Y = Y[denmask]
    Z = Z[denmask]
    T = T[denmask]
    Den = Den[denmask]

    Rad = Rad[denmask]
    Vol = Vol[denmask]
    Rad_den = np.multiply(Rad,Den)
    del Rad
#
    # Cross dot -----------------------------------------------------------------
    observers_xyz = hp.pix2vec(c.NSIDE, range(c.NPIX))
    # Line 17, * is matrix multiplication, ' is .T
    observers_xyz = np.array(observers_xyz).T
    cross_dot = np.matmul(observers_xyz,  observers_xyz.T)
    cross_dot[cross_dot<0] = 0
    cross_dot *= 4/c.NPIX

    # Tree ----------------------------------------------------------------------
    xyz = np.array([X, Y, Z]).T
    # Flux
    F_photo = np.zeros((c.NPIX, f_num))
    F_photo_temp = np.zeros((c.NPIX, f_num))
    # Dynamic Box -------------------------------------------------------------
    for i in tqdm(rays):
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

        # This naming is so bad
        r = np.logspace( -0.25, np.log10(rmax), N_ray)
        radii.append(r)
        alpha = (r[1] - r[0]) / (0.5 * ( r[0] + r[1]))
        dr = alpha * r

        x = r*mu_x
        y = r*mu_y
        z = r*mu_z
        xyz2 = np.array([x, y, z]).T
        del x, y, z
        
        #tree = eng.KDTreeSearcher(xyz, 'BucketSize', matlab.double([50]))
        #idx = eng.knnsearch(tree, xyz2, )
        # idx = [ int(idx[i][0] - 1) for i in range(len(idx))] # -1 because we start from 0
        tree = KDTree(xyz, leaf_size=50)
        _, idx = tree.query(xyz2, k=1)
        idx = [ int(idx[i][0]) for i in range(len(idx))] # -1 because we start from 0
        d = Den[idx] * c.den_converter
        t = T[idx]
        
        # for plot
        denplot.append(d)
        tplot.append(t)
        # Interpolate
        sigma_rossland = eng.interp2(T_cool2,Rho_cool2,rossland2,np.log(t),np.log(d),'linear',0)
        sigma_rossland = [sigma_rossland[0][i] for i in range(N_ray)]
        sigma_rossland_eval = np.exp(sigma_rossland) 
        mask = sigma_rossland_eval == np.exp(0)
        sigma_rossland_eval[mask] = 0
        # For abs
        sigma_plank = eng.interp2(T_cool2,Rho_cool2,plank2,np.log(t),np.log(d),'linear',0)
        sigma_plank = [sigma_plank[0][i] for i in range(N_ray)]
        sigma_plank_eval = np.exp(sigma_plank)
        mask = sigma_plank_eval == np.exp(0)
        sigma_plank_eval[mask] = 0
        # Do for scatter
        sigma_scatter = eng.interp2(T_cool2,Rho_cool2,scatter2,np.log(t),np.log(d),'linear',0)
        sigma_scatter = [sigma_scatter[0][i] for i in range(N_ray)]
        sigma_scatter_eval = np.exp(sigma_scatter)
        mask = sigma_scatter_eval == np.exp(0)
        sigma_scatter_eval[mask] = 0
        del sigma_rossland, sigma_plank, sigma_scatter
        gc.collect()
        
        # Optical Depth ---------------------------------------------------------------
        # Okay, line 232, this is the hard one.
        r_fuT = np.flipud(r.T)
        kappa_rossland = np.flipud(sigma_rossland_eval) 
        los = - np.flipud(sci.cumulative_trapezoid(kappa_rossland, r_fuT, initial = 0)) * c.Rsol_to_cm # dont know what it do but this is the conversion
        kappa_plank = np.flipud(sigma_plank_eval) 
        los_abs = - np.flipud(sci.cumulative_trapezoid(kappa_plank, r_fuT, initial = 0)) * c.Rsol_to_cm
        #k_effective = np.sqrt(3 * np.flipud(sigma_plank_eval) * np.flipud(sigma_rossland_eval)) 
        # So Elad says, try something different for opacity.
        k_effective = np.sqrt(3 * np.flipud(sigma_scatter_eval) * (np.flipud(sigma_scatter_eval) + np.flipud(sigma_plank_eval))) 

        los_effective = - np.flipud(sci.cumulative_trapezoid(k_effective, r_fuT, initial = 0)) * c.Rsol_to_cm
        
        # save for plot
        tau_eff.append(los_effective)
        tau_abs.append(los_abs)
        tau_ross.append(los)
        # Red -----------------------------------------------------------------------

        # Get 20 unique, nearest neighbors
        xyz3 = np.array([X[idx], Y[idx], Z[idx]]).T
        _, idxnew = tree.query(xyz3, k=20)
        # idxnew = [ int(idxnew[i][0]) for i in range(len(idxnew))] # -1 because we start from 0
        idxnew = np.unique(idxnew).T
        #idxnew = [ int(idxnew[i] -1) for i in range(len(idxnew))]
        # Cell radius
        dx = 0.5 * Vol[idx]**(1/3)
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
        gradz_m = np.nan_to_num(gradz_m, nan =  0)
        gradz = (gradz_p - gradz_m)/ (2*dx)
        del gradz_p, gradz_m

        grad = np.sqrt(gradx**2 + grady**2 + gradz**2)
        gradr = (mu_x * gradx) + (mu_y*grady) + (mu_z*gradz)
        del gradx, grady, gradz
        gc.collect()
        # v_grad = np.sqrt( (VX[idx] * gradx)**2 +  (VY[idx] * grady)**2 + (VZ[idx] * gradz)**2)
        
        R_lamda = grad / ( c.Rsol_to_cm * sigma_rossland_eval* Rad_den[idx])
        R_lamda[R_lamda < 1e-10] = 1e-10
        fld_factor = 3 * (1/np.tanh(R_lamda) - 1/R_lamda) / R_lamda 
        smoothed_flux = -uniform_filter1d(r.T**2 * fld_factor * gradr / sigma_rossland_eval, 7) # i have removed the minus
        # Red
        F_photo_temp = np.zeros((c.NPIX, f_num))
        b = np.where( ((smoothed_flux>0) & (los<2/3) ))[0][0] # elad_b = 3117
        Lphoto2 = 4*np.pi*c.c*smoothed_flux[b] * c.Msol_to_g / (c.t**2)
        EEr = Rad_den[idx]
        if Lphoto2 < 0:
            print('max length chosen')
            Lphoto2 = 1e100 # it means that it will always pick max_length for the negatives, maybe this is what we are getting wrong
        max_length = 4*np.pi*c.c*EEr[b]*r[b]**2 * c.Msol_to_g * c.Rsol_to_cm / (c.t**2)
        Lphoto = np.min( [Lphoto2, max_length])
        Lphoto_this[i] = Lphoto # save red
        del smoothed_flux, R_lamda, fld_factor, EEr, #los,
        gc.collect()
        # Spectra ---------------------------------------------------------------------
        los_effective[los_effective>30] = 30
        b2 = np.argmin(np.abs(los_effective-5))
        # elad_b2 = 3460
        # b2 = elad_b2
        
        # plt.figure(figsize = (3,3))
        # plt.plot(sigma_plank_eval, ':h', c = 'k')
        # plt.yscale('log')
        # plt.axvline(b2, c = 'purple', ls = '--', lw = 2)
        # plt.title(f'Obs: {i}')
        # plt.xlim(3000, 5000)
        # print('---')
        # print('Photosphere density', d[b], i)
        # print('Therm Radius density', d[b2], i)
        # print('Photosphere temperature', t[b], i)
        # print('Therm Radius temperature', t[b2], i)
        # print('Ross opacity', kappa_rossland[b])
        # print('Eff opacity', k_effective[b2])
        # print('Ross. opt depth', los[b], i)
        # print('Eff. opt depth', los_effective[b2], i)


        photosphere[i, idx_s] = r[b]
        therm_radius[i, idx_s] = r[b2]
        photo_d[i, idx_s] = d[b]
        photo_t[i, idx_s] = t[b2]
        therm_d[i, idx_s] = d[b]
        therm_t[i, idx_s] = t[b2]

        for k in range(b2, len(r)): 
            dr = r[k]-r[k-1]
            Vcell =  r[k]**2 * dr # there should be a (4 * np.pi / c.NPIX)*, but doesn't matter because we normalize
            wien = np.exp( c.h * frequencies / (c.Kb * t[k])) - 1 # Elad: min to avoid overflow
            black_body = frequencies**3 / (c.c**2 * wien)
            F_photo_temp[i,:] += sigma_plank_eval[k] * Vcell * np.exp(-los_effective[k]) * black_body # there should be a 4*np.pi*, but doesn't matter because we normalize
            # if k>b2+22 and k < b2+25:
            #     print(f'Lnu {F_photo_temp[i,446]:.2e}')
            #     print(f'Abs opac {sigma_plank_eval[k]:.2e}')
            #     print(f'T {t[k]:.2e}')
            #     print(f'rho {d[k]:.2e}')
            temperatures[k, i] = t[k]
        norm = Lphoto / np.trapz(F_photo_temp[i,:], frequencies)
        F_photo_temp[i,:] *= norm
        F_photo[i,:] = np.matmul(cross_dot[i,:], F_photo_temp[:,:])    
    
    for i in range(c.NPIX):
        Lphoto_all[idx_s] += Lphoto_this[i]
    Lphoto_all[idx_s] *= 4*np.pi/c.NPIX
#%%    
    plt.figure(figsize =(4,3))
    plt.title(f'Time: {day:.3f} $t_\mathrm{{FB}}$')
    plt.loglog(frequencies, frequencies * np.sum(F_photo, axis = 0), c='k')
    plt.loglog(frequencies, frequencies * F_photo[20], c='skyblue')
    plt.ylim(1e40, 1e45)
    plt.xlim(1e13, 1e20)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel(r'$\nu L_\nu$ [erg/s]')
    
    ypos = 3e44
    plt.axvspan(4e14, 7e14, alpha=0.2, color='gold')
    plt.text(3e14, ypos, 'Opt.', fontsize = 12)
    plt.axvspan(7e14, 7e16, alpha=0.2, color='purple')
    plt.text(5e15, ypos, 'UV', fontsize = 12)
    plt.axvspan(7e16, 5e18, alpha=0.2, color='cyan')
    plt.text(3e17, ypos, 'Soft X', fontsize = 12)
    plt.axvspan(5e18, 5e19, alpha=0.2, color='b')
    plt.text(5.8e18, ypos, 'Hard X', fontsize = 12)


#%%
# fig, ax = plt.subplots(1,1, figsize = (4,4))
# Rt = 0.47 * (1e5/0.5)**(1/3)
# idx = 2
# ax.plot(radii[0]/Rt, tau_eff[idx], c = c.darkb, label = 'Eff.')
# ax.plot(radii[0]/Rt, tau_abs[idx], c = c.prasinaki, ls = ':', label = 'Abs.')
# ax.plot(radii[0]/Rt, tau_ross[idx], c = c.kroki, ls = '--', label = 'Ross.')
# ax2 = ax.twinx()
# ax2.plot(radii[0]/Rt, tplot[idx], c=c.AEK)
# # ax3 = ax2.twinx()
# # ax3.loglog(radii[0]/Rt, denplot[idx], c='k')


# # R-lines
# # ax.text(1.1, 1e-18, r'$R_\mathrm{T}$', c = 'b',)
# # ax.text(0.2, 1e-18, r'$R_\mathrm{soft}$', c = 'r')
# # ax.axvline(Rt/Rt, c = 'b', ls = ':')
# # ax.axvline(0.6 * Rt/Rt, c = 'r', ls = ':')

# # Special lines
# # ax.text(1.1, 1e-19, r'Photo', c = 'g',)
# # ax.text(0.2, 1e-19, r'Therm', c = 'purple')
# # ax.axvline(photosphere[rays[idx]]/Rt, c = 'g', ls = '-', lw = 2, 
# #            label = 'Photo')
# ax.axvline(therm_radius[rays[idx]][0]/Rt, c = 'purple', ls = '--', lw = 2,
#            label = 'Therm')

# # Scales & Labels
# ax.set_xlim(1e1, 1e2)
# # ax.set_ylim(1e-20, 1e-16)
# #ax.set_xscale('log')
# ax.set_ylabel('Optical Depth')
# ax.set_ylim(0, 6)
# #ax.set_yscale('log')
# ax2.set_yscale('log')
# ax.set_xlabel(r'r [r/$R_\mathrm{T}$]')
# #ax2.set_ylabel(r'Density [g/cm$^3$]')
# ax2.set_ylabel(r'Temperatures [K]')

# ax.legend(loc = 'lower right')
#     #%% Save data ------------------------------------------------------------------
#     if save:
#         if alice:
#             pre_saving = '/home/s3745597/data1/TDE/tde_comparison/data/'
#         else:
#             pre_saving = 'data/blue'
        
#         np.savetxt(f'{pre_saving}blue/{sim}/freqs.txt', frequencies)
#         np.savetxt(f'{pre_saving}blue/{sim}/spectra{snap}.txt', F_photo)
#         np.savetxt(f'{pre_saving}blue/{sim}/photosphere{snap}.txt', photosphere)
#         np.savetxt(f'{pre_saving}blue/{sim}/thermradius{snap}.txt', therm_radius)
#         np.savetxt(f'{pre_saving}blue/{sim}/temperaturemap{snap}.txt', temperatures)

# eng.exit()


