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

from src.Opacity.LTE_loader import T_opac_ex, Rho_opac_ex, rossland_ex, plank_ex, scattering_ex
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
    ms = [4, 5, 6]
    ms = [5]
    mstar = 0.5
    rstar = 0.47
def masker(mask, list_of_quantities):
    new_list = []
    for quantity in list_of_quantities:
        new_list.append(quantity[mask])
    return (*new_list,)

for m in ms:
    if m == 4:
        fixes = [116, 136, 164, 179, 199, 218, 240, 272, 297, 300, 348]
        fixes = [348]
    if m == 5:
        fixes = [227, 236, 288, 301, 308, 349]
        fixes = [349]
    if m == 6:
        fixes = [180, 290, 315, 325, 351, 379, 444]
        fixes = [444]
        
    Rt = rstar * (10**m/mstar)**(1/3)
    for fix in tqdm(fixes):
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
        # observers = [92,]
        x_photo = np.zeros(len(observers))
        y_photo = np.zeros(len(observers))
        pre = 'data/bluepaper/rays'
        os.system(f'mkdir {pre}{m}{fix}')
        los_es = []
        rs = []
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
            los_es.append(los)
            photosphere = np.where(los < 2/3)[0][0]
            
            rs.append(r)
            Rphs.append(r[photosphere])
            x_photo[i] = ray_x[photosphere]
            y_photo[i] = ray_y[photosphere]
            
            # Ray plot
            # plt.figure(figsize = (5,5))
            # plt.axhline(5802, c = 'navy', ls = '--')
            # plt.plot(r/amin, t, '-o', c = 'royalblue', lw = 0.75, markersize = 1.2,
            #         label = 'T [K]')
            # plt.axhline(58002693, c = 'navy', ls = '--', label = r'T table limit')
        
            # plt.axhline(100, c = 'darkorange', ls = '--')
            # plt.plot(r/amin, d, '-o', c = 'peru', lw = 0.75, markersize = 1.2,
            #         label = r'$\rho$ [g/cm$^3$]')
            # plt.axhline(9.99e-11, c = 'darkorange', ls = '--', label = r'$\rho$ table limit')
            
            # plt.plot(r/amin, sigma_rossland_eval, '-o', c = 'r', lw = 0.75, markersize = 1.2,
            #         label = r'$\alpha_\mathrm{ross}$ [1/cm]')
            
            # plt.plot(r/amin, delta_r * c.Rsol_to_cm, '-o', c = 'hotpink', 
            #         lw = 0.75, markersize = 1.2, label = 'Cell Size [cm]')
        
            
        #%% Taus plot
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib.axes as maxes

        import colorcet
        gs = gridspec.GridSpec(3, 3, )  # 1/3 and 2/3 height split
        fig = plt.figure(figsize=(8, 8))
        fig.suptitle(f'$10^{m}$ M$_\odot$ - Snapshot {fix}', fontsize = 18)
        taup = fig.add_subplot(gs[0:2, 0:3])
        obs = fig.add_subplot(gs[-1:, 0:1])
        proj = fig.add_subplot(gs[-1:, 1:])
        observer_labels = ['88 x', '89', '90', '91', '92 y', '93', '94 ', '95', 
                           '96 -x',
                           '97 ', '98', '99', '100 -y', '101', '102', '103 x']
        for r, los, color, label  in zip(rs, los_es, c.r16_palette, observer_labels):            
            taup.plot(r/amin, los, '-o', c = color, lw = 2, markersize = 0.1,
                     label = f'{label}')
        taup.axvline(Rt/amin, c = 'gray', ls = ':', label = r'$R_\mathrm{T}$')
        taup.axhline(2/3, c = 'r', ls = ':', label = '2/3', zorder = 11)
        avg_photo = np.mean(Rphs)
        taup.axvline(avg_photo/amin, ls = '-.', c = c.AEK, 
                    label = 'Avg. Photosphere', zorder = 9)
        taup.legend(loc = 'lower left',  ncols = 3,
                   fontsize = 12, frameon = False)
        taup.set_ylabel(r'$\tau = \int^\infty_r \alpha_\mathrm{ross} dr''$', 
                        fontsize = 15)
        taup.set_xlabel(r'Radial Coordinate [$\alpha_\mathrm{min}$]', 
                        fontsize = 15)
        taup.loglog()
        taup.set_xlim(1e-3, 1e2)
        
        # Photosphere plot
        x_photo = np.append(x_photo, x_photo[0])
        y_photo = np.append(y_photo, y_photo[0])
    
        obs.set_xlabel(r'Observers', fontsize = 12)
        obs.set_ylabel(r'$R_\mathrm{photo}$ [$\alpha_\mathrm{min}$]', 
                       fontsize = 12)
        step = 1
        orbplanemask = np.abs(Z) < Vol**(1/3)
        obs.plot(observers, np.array(Rphs)/amin, '-o', c = 'k', 
                   lw = 0.75, markersize = 0.1)
        img = proj.scatter(X[orbplanemask][::step] / amin, Y[orbplanemask][::step] / amin,
                            s = 0.1,
                            vmin = -16, vmax = -8,
                            c = np.log10(Den[orbplanemask][::step]), 
                            cmap = 'cet_fire')
        # Nice colovar
        divider = make_axes_locatable(proj)
        cax = divider.append_axes('right', size = '4%', pad = 0.05,
                                  axes_class = maxes.Axes)
        cb = plt.colorbar(img, cax = cax, orientation = 'vertical')
        cb.set_label(r'$log \rho$ [g/cm$^3$]', fontsize = 12)
        
        # Do line for phot
        proj.plot(x_photo / amin, y_photo / amin, '-', c = 'k', lw = 1, zorder = 1)
        observer_labels = ['88', '89', '90', '91', '92', '93', '94', '95', '96',
                           '97', '98', '99', '100', '101', '102', '103']
        for x, y, rph, color, label in zip(x_photo, y_photo, Rphs, c.r16_palette, 
                                         observer_labels, ):
            obs.plot(int(label), rph/amin, '-o', color = color , 
                       lw = 0.75, markersize = 5)
            
            proj.scatter(x / amin, y / amin, c = color, 
                          marker = f'${int(label)}$', linewidths = 1,
                          s = 75, zorder = 3)
        proj.set_xlim(-6.5, 2.5)
        proj.set_ylim(-2, 2)
        
        proj.set_xlabel(r'X coordinate [$\alpha_\mathrm{min}]$', fontsize = 12)
        proj.set_ylabel(r'Y coordinate [$\alpha_\mathrm{min}]$', fontsize = 12)
        plt.tight_layout()
        plt.savefig(f'{pre}{m}{fix}/{m}_{fix}_tau_obs_proj.png')
        
