#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 09:16:59 2025

@author: konstantinos
"""

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
from tqdm import tqdm

from src.Opacity.LTE_loader import T_opac_ex, Rho_opac_ex, rossland_ex, plank_ex, scattering_ex
import src.Utilities.prelude as c
from src.Utilities.parser import parse
from src.Utilities.isalice import isalice
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
    ms = [6]
    mstar = 0.5
    rstar = 0.47

for m in ms:
    pre = f'{m}/'
    if m == 4:
        fixes = [179, 240, 300] 
    elif m == 5:
        fixes = [227, 288, 349]
    elif m == 6:
        fixes = [315, 379, 444] # 420->444
        fixes = [444]
    for fix in fixes:
        Rt = rstar * (10**m/mstar)**(1/3)

        X = np.load(f'{pre}{fix}/CMx_{fix}.npy')
        Y = np.load(f'{pre}{fix}/CMy_{fix}.npy')
        Z = np.load(f'{pre}{fix}/CMz_{fix}.npy')
        T = np.load(f'{pre}{fix}/T_{fix}.npy')
        Den = np.load(f'{pre}{fix}/Den_{fix}.npy')
        Rad = np.load(f'{pre}{fix}/Rad_{fix}.npy')
        Vol = np.load(f'{pre}{fix}/Vol_{fix}.npy')
        box = np.load(f'{pre}{fix}/box_{fix}.npy')
        day = np.loadtxt(f'{pre}{fix}/tbytfb_{fix}.txt')
        
        # lowrho = Den < 1e-16
        # X = X[lowrho]
        # Y = Y[lowrho]
        # Z = Z[lowrho]
        # T = T[lowrho]
        # Den = Den[lowrho]
        # Rad = Rad[lowrho]
        # Vol = Vol[lowrho]
        
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
        scattering2 = scattering_ex
        
        ### Do it --- --- ---
        xyz = np.array([X, Y, Z]).T
        
        # Flux
        F_photo = np.zeros((c.NPIX, f_num))
        F_photo_temp = np.zeros((c.NPIX, f_num))
        photosphere = []
        colorsphere = []
        time_start = 0
        reds = np.zeros(c.NPIX)
        
        # Iterate over observers
        zsweep = [104, 136, 152, 167, 168, 179, 180, 187, 188, 191,]# 140]
        walls = np.zeros(len(zsweep))
        bubbles = np.zeros(len(zsweep))

        #zsweep = [136]
        zsweep_nside2 = [ 20, 27, 28, 29, 35, 36, 43, 37, 42, 44, 47 ]
        # zsweep = [72, 80, 76, ] # 84, 188, 0]
        # zsweep_nside2 = [47]
        wall_counter = 0
        for i in tqdm(range(109, 111,1)): #c.NPIX, 10)):
            # Progress 
            # time_end = time.time()
            # print(f'Snap: {fix}, Obs: {i}', 
            #       flush=False)
            # print(f'Time for prev. Obs: {(time_end - time_start)/60} min', 
            #       flush = False)
            # time_start = time.time()
            # sys.stdout.flush()
            
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
            # rmax = 5*amin
            rmin = -0.25
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
            
            sigma_plank = eng.interp2(T_cool2,Rho_cool2,plank2.T, 
                                      np.log(t),np.log(d),'linear',0)
            sigma_plank = np.array(sigma_plank)[0]
            sigma_plank_eval = np.exp(sigma_plank)
            
            sigma_scattering = eng.interp2(T_cool2,Rho_cool2,scattering2.T, 
                                      np.log(t),np.log(d),'linear',0)
            sigma_scattering = np.array(sigma_scattering)[0]

            sigma_scattering_eval = np.exp(sigma_scattering)
            
            # Check that we didnt underflow
            plank_is_ok_mask = sigma_plank_eval != 1.0
            scatter_is_ok_mask = sigma_scattering_eval != 1.0
            both_are_ok_mask = plank_is_ok_mask * scatter_is_ok_mask
            sigma_plank_eval = sigma_plank_eval[both_are_ok_mask]
            sigma_scattering_eval = sigma_scattering_eval[both_are_ok_mask]
            r = r[both_are_ok_mask]
            t = t[both_are_ok_mask]
            d = d[both_are_ok_mask]
            sigma_rossland_eval = sigma_rossland_eval[both_are_ok_mask]
            del sigma_rossland, sigma_plank 
            gc.collect()
            # Optical Depth ---.    
            r_fuT = np.flipud(r.T)
            # kappa_rossland = np.flipud(sigma_rossland_eval) 
            # los = - np.flipud(sci.cumulative_trapezoid(kappa_rossland, r_fuT, initial = 0)) * c.Rsol_to_cm # dont know what it do but this is the conversion
            
            # kappa_plank = np.flipud(sigma_plank_eval) 
            # los_abs = - np.flipud(sci.cumulative_trapezoid(kappa_plank, r_fuT, initial = 0)) * c.Rsol_to_cm
            # k_effective = np.sqrt(3 * np.flipud(sigma_plank_eval) * np.flipud(sigma_rossland_eval)) 
            # los_effective = - np.flipud(sci.cumulative_trapezoid(k_effective, r_fuT, initial = 0)) * c.Rsol_to_cm
            
            # Try out elads scattering + rosseland
            kappa_rossland = np.flipud(sigma_scattering_eval) + np.flipud(sigma_plank_eval)
            los = - np.flipud(sci.cumulative_trapezoid(kappa_rossland, r_fuT, initial = 0)) * c.Rsol_to_cm # dont know what it do but this is the conversion
            
            kappa_plank = np.flipud(sigma_plank_eval) 
            los_abs = - np.flipud(sci.cumulative_trapezoid(kappa_plank, r_fuT, initial = 0)) * c.Rsol_to_cm
            k_effective = np.sqrt(3 * np.flipud(sigma_plank_eval) * kappa_rossland) 
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
            
            grad = grad[both_are_ok_mask]
            R_lamda = grad / ( c.Rsol_to_cm * sigma_rossland_eval* Rad_den[idx][both_are_ok_mask])
            R_lamda[R_lamda < 1e-10] = 1e-10
            fld_factor = 3 * (1/np.tanh(R_lamda) - 1/R_lamda) / R_lamda 
            
            gradr = gradr[both_are_ok_mask]
            # fld_factor = fld_factor[both_are_ok_mask]
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
            EEr = EEr[both_are_ok_mask]
            if Lphoto2 < 0:
                Lphoto2 = 1e100 # it means that it will always pick max_length for the negatives, maybe this is what we are getting wrong
            max_length = 4*np.pi*c.c*EEr[b]*r[b]**2 * c.Msol_to_g * c.Rsol_to_cm / (c.t**2)
            reds[i] = np.min( [Lphoto2, max_length])
            del smoothed_flux, R_lamda, fld_factor, EEr, los,
            gc.collect()
            # Spectra ---
            if los_effective[b2-1] >= 29:
                wall_cell  = 4 * np.pi * c.stefan * t[b2-1]**4 * ( r[b2-1] * c.Rsol_to_cm)**2
                bubble_temp = 0
                for k in range(b2-1, len(r)):
                    dr = r[k]-r[k-1]
                    Vcell =  r[k]**2 * dr # there should be a (4 * np.pi / 192)*, but doesn't matter because we normalize
                    wien = np.exp(c.h * frequencies / (c.kb * t[k])) - 1
                    black_body = frequencies**3 / (c.c**2 * wien)
                    if k == b2-1:
                        # wall_norm = 
                        F_photo_temp[i,:] += wall_cell* black_body / np.trapz(black_body, frequencies)
                    else:
                        bubble_temp += sigma_plank_eval[k] * Vcell * np.exp(-los_effective[k]) * black_body
                        F_photo_temp[i,:] += bubble_temp
                # bubble = np.trapz(bubble_temp, frequencies)
                # bubbles[wall_counter] = bubble
                # walls[wall_counter] = wall_cell
                # wall_counter += 1
                # print('hi')
                    # if t[k] > 1e8:
                    #     bubble += c.c * sigma_plank_eval[k] * Vcell * np.exp(-los_effective[k]) * black_body
            else:
                for k in range(b2, len(r)):
                    dr = r[k]-r[k-1]
                    Vcell =  r[k]**2 * dr # there should be a (4 * np.pi / 192)*, but doesn't matter because we normalize
                    wien = np.exp(c.h * frequencies / (c.kb * t[k])) - 1
                    black_body = frequencies**3 / (c.c**2 * wien)
                    F_photo_temp[i,:] += sigma_plank_eval[k] * Vcell * np.exp(-los_effective[k]) * black_body
        
            norm = reds[i] / np.trapz(F_photo_temp[i,:], frequencies)
            F_photo_temp[i,:] *= norm
            F_photo[i,:] = np.dot(cross_dot[i,:], F_photo_temp)   
            #
            import matplotlib.pyplot as plt
            plt.ioff()
            plt.figure()
            plt.plot(r / amin, np.log10(t), '-o', c='k', 
                     lw = 1.3, markersize = 0.4, label = 'logT [K]', zorder = 10)
            plt.plot(r/amin, np.log10( np.sqrt(3 * sigma_plank_eval * (sigma_plank_eval + sigma_scattering_eval))), 
                     '-o', c='skyblue', lw = 0.3, markersize = 0.4, 
                     label = r'$\sqrt{ 3\sigma_\mathrm{abs} \left( \sigma_\mathrm{sca} + \sigma_\mathrm{abs} \right)}$')
            plt.plot(r/amin, np.log10(sigma_scattering_eval + sigma_plank_eval), 
                     '-o', c='r', lw = 0.3, markersize = 0.4, 
                      label = r'$\sigma_\mathrm{sca} + \sigma_\mathrm{abs}$')
            plt.plot(r / amin, np.log10(d * c.den_converter), 
                     '-o', c='darkorange', lw = 0.3, markersize = 0.4, 
                     label = r'log $\rho$ [cgs]')
            plt.plot(r/ amin, los_effective, c = 'darkgreen', label = 'tau effective')
            # plt.axhline(1, c = 'tomato', ls = ':')
            # plt.axhline(5, c = 'maroon', ls = ':')
            plt.axhline(7.7634, c = 'r', ls = '--')
            plt.axhline(np.log10(5802.243894044859), c = 'r', ls = '--', label = 'table edge')
            plt.axvline(r[b] / amin, 
                        c = c.AEK, ls = '--', label = 'photosphere')
            plt.axvline(r[b2] / amin, 
                        c = 'slateblue', ls = '--', label = 'colorsphere')
            plt.legend(fontsize =5, frameon = False)
            plt.xscale('log')
            plt.xlabel('r [amin]')
            plt.ylabel('logT [K]')
            plt.title(f'MBH {m} - {fix} - Obs {i}')
            plt.ylim(-19, 10)
            plt.savefig(f'data/bluepaper/{m}ray{fix}{i}.png')
        ### Bolometric ---
        red = 4 * np.pi * np.mean(reds) # this 4pi here shouldn't exist, leaving it for posterity
        
        # Saving ---
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
eng.exit()
