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

from src.Opacity.LTE_loader import T_opac_ex, Rho_opac_ex, rossland_ex, plank_ex, scattering_ex
import src.Utilities.prelude as c
from src.Utilities.parser import parse
from src.Utilities.isalice import isalice
from src.Utilities.loaders import local_loader, boxer

alice, plot = isalice()

save = True # WATCH OUT    WATCH OUT    WATCH OUT    WATCH OUT    WATCH OUT
plt.ioff()
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
    ms = [4, 5, 6]
    ms = [6]
    mstar = 0.5
    rstar = 0.47

for m in ms:
    if m == 4:
        fixes = [116, 136, 164, 179, 199, 218, 240, 272, 297, 300, 348]
        fixes = [348]
    if m == 5:
        fixes = [227, 236, 288, 301, 308, 349]
    if m == 6:
        fixes = [180, 290, 315, 325, 351, 379, 444]
        fixes = [444]
        
    Rt = rstar * (10**m/mstar)**(1/3)
    for fix in tqdm(fixes):
        try:
            X, Y, Z, Den, T, Rad, Vol, box, day = local_loader(m, fix,                                                        'thermodynamics')
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
        T_cool2 = T_opac_ex
        Rho_cool2 = Rho_opac_ex
        rossland2 = rossland_ex
        plank2 = plank_ex
        scattering2 = scattering_ex
        
        ### Do it --- --- ---
        xyz = np.array([X, Y, Z]).T
        
        # Make the ray
        rmin = -0.25
        Rphs = []
        # observers = np.linspace(0,c.NPIX - 1,5)
        observers = np.arange(88, 104)[::1] # Equatorial
        # observers = [88,]
        x_photo = np.zeros(len(observers))
        y_photo = np.zeros(len(observers))
        pre = 'data/bluepaper/rays'
        os.system(f'mkdir {pre}{m}{fix}')
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
            
            # Interpolate ---
            sigma_rossland = eng.interp2(T_cool2,Rho_cool2,rossland2.T # needs T for the new RICH extrapol
                                         ,np.log(t), np.log(d),'linear',0)
            sigma_rossland = np.array(sigma_rossland)[0]
            # sigma_rossland = sigma_rossland[sigma_rossland != 1.0] 
            sigma_rossland_eval = np.exp(sigma_rossland) 
            
            
            # Optical Depth --
            delta_r = np.diff(r)
            delta_r = np.insert(delta_r, 0, delta_r[0]) # len -1 -> len
            delta_taus = r * c.Rsol_to_cm * sigma_rossland_eval
            
            # Photosphere finding
            interiormask = r < 2*Rt # dont have it be the first bunch of cells
            modified_delta_taus = copy.deepcopy(delta_taus)
            modified_delta_taus[interiormask] = 1000# arbitrary to trace
            
            # Red ---
            # Get 20 unique, nearest neighbors
            # xyz3 = np.array([X[idx], Y[idx], Z[idx]]).T
            # xyz3 = np.array([X[idx], Y[idx], Z[idx]]).T
            # _, idxnew = tree.query(xyz3, k=20)
            # idxnew = np.unique(idxnew).T
        
            # # Cell radius
            # dx = 0.5 * Vol[idx]**(1/3)
            
            # # Get the Grads    
            # f_inter_input = np.array([ X[idxnew], Y[idxnew], Z[idxnew] ]).T
        
            # gradx_p = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
            #                     xi = np.array([ X[idx]+dx, Y[idx], Z[idx]]).T )
            # gradx_m = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
            #                     xi = np.array([ X[idx]-dx, Y[idx], Z[idx]]).T )
            # gradx = (gradx_p - gradx_m)/ (2*dx)
            # gradx = np.nan_to_num(gradx, nan =  0)
            # del gradx_p, gradx_m
        
            # grady_p = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
            #                     xi = np.array([ X[idx], Y[idx]+dx, Z[idx]]).T )
            # grady_m = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
            #                     xi = np.array([ X[idx], Y[idx]-dx, Z[idx]]).T )
            # grady = (grady_p - grady_m)/ (2*dx)
            # grady = np.nan_to_num(grady, nan =  0)
            # del grady_p, grady_m
        
            # gradz_p = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
            #                     xi = np.array([ X[idx], Y[idx], Z[idx]+dx]).T )
            # gradz_m = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
            #                     xi = np.array([ X[idx], Y[idx], Z[idx]-dx]).T )
            # gradz_m = np.nan_to_num(gradz_m, nan =  0)
            # gradz = (gradz_p - gradz_m)/ (2*dx)
            # del gradz_p, gradz_m
        
            # grad = np.sqrt(gradx**2 + grady**2 + gradz**2)
            # gradr = (mu_x * gradx) + (mu_y*grady) + (mu_z*gradz)
            # del gradx, grady, gradz
            # gc.collect()
            
            # R_lamda = grad / ( c.Rsol_to_cm * sigma_rossland_eval* Rad_den[idx])
            # R_lamda[R_lamda < 1e-10] = 1e-10
            # fld_factor = 3 * (1/np.tanh(R_lamda) - 1/R_lamda) / R_lamda 
            
            # # fld_factor = fld_factor[both_are_ok_mask]
            # smoothed_flux = -uniform_filter1d(r.T**2 * fld_factor * gradr / sigma_rossland_eval, 7) 
            
            
            # posflux = smoothed_flux > 0
            # modified_delta_taus[~posflux] = 1000
            photosphere = np.argmax(modified_delta_taus < 2/3) - 1 
            
            # Gap jumper
            nearby_tau = np.mean(modified_delta_taus[photosphere - 3: photosphere + 3])
            threshold = 10
            if nearby_tau > threshold * modified_delta_taus[photosphere+1]:
                print('trying to jump...')
                # criterion = 0
                # while np.abs(criterion) > 0.1:
                smaller_than_gap = r < r[photosphere+10]
                modified_delta_taus[smaller_than_gap] = 1000
                photosphere = np.argmax(modified_delta_taus < 2/3) - 1 
                    # criterion = modified_delta_taus[photosphere-1] - modified_delta_taus[photosphere+1]


            Rphs.append(r[photosphere])
            x_photo[i] = ray_x[photosphere]
            y_photo[i] = ray_y[photosphere]
            
            
            # Ray plot
            plt.figure(figsize = (5,5))
            plt.axvline(Rt/amin, c = 'gray', label = r'$R_\mathrm{T}$')

        

            
            plt.axhline(5802, c = 'navy', ls = '--')
            plt.plot(r/amin, t, '-o', c = 'royalblue', lw = 0.75, markersize = 1.2,
                    label = 'T [K]')
            plt.axhline(58002693, c = 'navy', ls = '--', label = r'T table limit')
        
            plt.axhline(100, c = 'darkorange', ls = '--')
            plt.plot(r/amin, d, '-o', c = 'peru', lw = 0.75, markersize = 1.2,
                    label = r'$\rho$ [g/cm$^3$]')
            plt.axhline(9.99e-11, c = 'darkorange', ls = '--', label = r'$\rho$ table limit')
            
            plt.plot(r/amin, sigma_rossland_eval, '-o', c = 'r', lw = 0.75, markersize = 1.2,
                    label = r'$\alpha_\mathrm{ross}$ [1/cm]')
            
            plt.plot(r/amin, delta_r * c.Rsol_to_cm, '-o', c = 'hotpink', 
                    lw = 0.75, markersize = 1.2, label = 'Cell Size [cm]')
            
            plt.plot(r/amin, delta_taus, '-o', c = 'k', lw = 0.75, markersize = 1.2,
                    label = r'$\Delta \tau = \kappa \rho \Delta r$', zorder = 10)
            plt.axhline(2/3, c = 'olive', ls = ':', label = '2/3', zorder = 11)
            plt.axvline(r[photosphere]/amin, ls = '-.', c = c.AEK, 
                        label = 'Photosphere', zorder = 9)
            
            plt.xlabel(r'Radial Coordinate [$\alpha_\mathrm{min}$]')
            # plt.ylabel(r'Fractional optical depth $\Delta \tau = \kappa \rho \Delta r$')
            plt.ylabel('stuff')
            plt.loglog()
            plt.xlim(1e-3, 1e2)
            plt.legend(bbox_to_anchor = [1.25, 0.1, 0.1, 0.9], 
                       fontsize = 8, frameon = False)
            plt.title(f'BH: {m} Snapshot {fix} Observer {obs}')
            plt.savefig(f'{pre}{m}{fix}/{obs}.png', bbox_inches = 'tight')
            
            ## Big drop method
            # relative_tau = np.diff(delta_taus) / delta_taus[:-1]
            # threshold = -0.5  # Define what counts as a "large" drop
            #drop_index = np.where(dy < threshold)[0][0]  # Get the first occurrence
        x_photo = np.append(x_photo, x_photo[0])
        y_photo = np.append(y_photo, y_photo[0])
    
        # fig, _ = plt.subplots(1,2,figsize = (8,4), tight_layout = True)
        fig = plt.figure(figsize=(8, 4))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])  # 1/3 and 2/3 height split
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1:])
        ax = (ax1, ax2)
        fig.suptitle(f'BH: {m} Snapshot {fix}')
        ax[0].plot(observers, np.array(Rphs)/amin, '-o', c = 'k' , lw = 0.75, markersize = 2)
        ax[0].set_xlabel(r'Observers')
        ax[0].set_ylabel(r'$R_\mathrm{photo}$ [$\alpha_\mathrm{min}$]')
        step = 1
        orbplanemask = np.abs(Z) < Vol**(1/3)
        import colorcet
        
        img = ax[1].scatter(X[orbplanemask][::step] / amin, Y[orbplanemask][::step] / amin,
                            s = 0.1,
                            vmin = -14, vmax = -8,
                            c = np.log10(Den[orbplanemask][::step]), 
                            cmap = 'cet_fire')
        cb = plt.colorbar(img)
        cb.set_label(r'$log \rho$ [g/cm$^3$]')
        
        ax[1].plot(x_photo / amin, y_photo / amin, '-o', c = c.cyan, 
                   markersize = 3, lw = 2, 
                   label = 'Equatorial \n Photosphere')
        ax[1].set_xlim(-8, 3)
        ax[1].set_ylim(-8, 3)
        
        ax[1].set_xlabel(r'X coordinate [$\alpha_\mathrm{min}]$')
        ax[1].set_ylabel(r'Y coordinate [$\alpha_\mathrm{min}]$')
        ax[1].legend(frameon = True)
        plt.tight_layout()
        plt.savefig(f'{pre}{m}{fix}/proj.png')
        
