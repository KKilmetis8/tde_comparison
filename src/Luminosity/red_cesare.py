#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 16:42:47 2023

@author: paola

Equations refer to Krumholtz '07

NOTES FOR OTHERS:
- make changes in variables: m (power index of the BB mass), 
fixes (number of snapshots) anf thus days
"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

# Vanilla Imports
import numpy as np
import matplotlib.pyplot as plt
import numba
from datetime import datetime
# Custom Imports
from src.Opacity.opacity_table import opacity
from src.Calculators.ray_cesare import ray_maker
from src.Luminosity.special_radii import calc_photosphere
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [5 , 4]

#%% Constants & Converter
today = datetime.now()
c_cgs = 3e10 # [cm/s]
Rsol_to_cm = 6.957e10 # [cm]
alice = False
#%%
##
# FUNCTIONS
##
###
def spacing(t):
    if alice:
        start = 1.00325
        end = 1.3
        n_start = 700 #850
        n_end = 3000
    else:
        start = 1
        end = 1.3
        n_start = 700 #850
        n_end = 3000
    n = (t-start) * n_end - (t - end) * n_start
    n /= (end - start)
    return n

def select_fix(m):
    if m == 4:
        snapshots = [233] #, 254, 263, 277 , 293, 308, 322]
        days = [1]# , 1.2, 1.3, 1.4, 1.56, 1.7, 1.8] 
    if m == 6:
        if alice:
            snapshots = np.arange(844, 1008 + 1)
            days = [1.00325,1.007,1.01075,1.01425,1.018,1.02175,1.0255,1.029,1.03275,1.0365,1.04025,1.04375,1.0475,1.05125,1.055,1.0585,1.06225,1.066,1.06975,1.07325,1.077,1.08075,1.0845,1.088,1.09175,1.0955,1.09925,1.10275,1.1065,1.11025,1.114,1.1175,1.12125,1.125,1.12875,1.13225,1.136,1.13975,1.1435,1.147,1.15075,1.1545,1.15825,1.16175,1.1655,1.16925,1.173,1.1765,1.18025,1.184,1.18775,1.19125,1.195,1.19875,1.2025,1.206,1.20975,1.2135,1.21725,1.22075,1.2245,1.22825,1.232,1.2355,1.23925,1.243,1.24675,1.25025,1.254,1.25775,1.2615,1.265,1.26875,1.2725,1.27625,1.27975,1.2835,1.28725,1.291,1.2945,1.29825,1.302,1.30575,1.30925,1.313,1.31675,1.3205,1.324,1.32775,1.3315,1.33525,1.33875,1.3425,1.34625,1.35,1.3535,1.35725,1.361,1.36475,1.36825,1.372,1.37575,1.3795,1.383,1.38675,1.3905,1.39425,1.39775,1.4015,1.40525,1.409,1.4125,1.41625,1.42,1.42375,1.42725,1.431,1.43475,1.4385,1.442,1.44575,1.4495,1.45325,1.45675,1.4605,1.46425,1.468,1.4715,1.47525,1.479,1.48275,1.48625,1.49,1.49375,1.4975,1.501,1.50475,1.5085,1.51225,1.51575,1.5195,1.52325,1.527,1.5305,1.53425,1.538,1.54175,1.54525,1.549,1.55275,1.5565,1.56,1.56375,1.5675,1.57125,1.57475,1.5785,1.58225,1.586,1.5895,1.59325,1.597,1.60075,1.60425,1.608]
        else:
            snapshots = [844, 881, 925, 950] #[844, 881, 882, 898, 925, 950]
            days = [1, 1.1, 1.3, 1.4] #[1, 1.139, 1.143, 1.2, 1.3, 1.4] # t/t_fb
            
        #     const = 0.05
    #     beginning = 1200
    # num_array = beginning * np.ones(len(snapshots))
    # for i in range(1,len(num_array)):
    #         num_array[i] = int(1.5 * num_array[i-1])
        num_array = [spacing(d) for d in days]
    return snapshots, days, num_array

@numba.njit
def grad_calculator(ray: np.array, radii: np.array, sphere_radius: int): 
    # For a single ray (in logspace) get 
    # the index of radius closest to sphere_radius and the gradE there.
    # Everything is in CGS.
    for i, radius in enumerate(radii):
        if radius > sphere_radius:
            idx = i - 1 
            break
        
    step = radii[idx+1] - radii[idx]
    
    grad_E = (ray[idx+1] - ray[idx]) / step 

    return grad_E, idx

    
def flux_calculator(grad_E, idx_tot, 
                    rays, rays_T, rays_den):
    """
    Get the flux for every observer.
    Eevrything is in CGS.

    Parameters: 
    grad_E idx_tot are 1D-array of lenght = len(rays)
    rays, rays_T, rays_den are len(rays) x N_cells arrays
    """
    f = np.zeros(len(grad_E))
    max_count = 0
    max_but_zero_count = 0
    zero_count = 0
    flux_zero = 0
    flux_count = 0
    for i, ray in enumerate(rays):
        # We compute stuff OUTSIDE the photosphere
        # (which is at index idx_tot[i])
        idx = idx_tot[i]+1 #  
        Energy = ray[idx]
        max_travel = np.sign(-grad_E[i]) * c_cgs * Energy # or should we keep the abs???
        
        Temperature = rays_T[i][idx]
        Density = rays_den[i][idx]
        
        # Ensure we can interpolate
        rho_low = np.exp(-45)
        T_low = np.exp(8.77)
        T_high = np.exp(17.8)
        
        # If here is nothing, light continues
        if Density < rho_low:
            max_count += 1
            f[i] = max_travel
            if max_travel == 0:
                max_but_zero_count +=1
            continue
        
        # If stream, no light 
        if Temperature < T_low: 
            zero_count += 1
            f[i] = 0 
            continue
        
        # T too high => scattering
        if Temperature > T_high:
            Tscatter = np.exp(17.87)
            k_ross = opacity(Tscatter, Density, 'scattering', ln = False)
        else:    
            # Get Opacity, NOTE: Breaks Numba
            k_ross = opacity(Temperature, Density, 'rosseland', ln = False)
        
        # Calc R, eq. 28
        R = np.abs(grad_E[i]) /  (k_ross * Energy)
        invR = 1 / R
        
        # Calc lambda, eq. 27
        coth = 1 / np.tanh(R)
        lamda = invR * (coth - invR)
        # Calc Flux, eq. 26
        Flux = - c_cgs * grad_E[i]  * lamda / k_ross
        
        # Choose
        if Flux > max_travel:
            f[i] = max_travel
            max_count += 1
            if max_travel == 0:
                max_but_zero_count +=1
        else:
            flux_count += 1
            f[i] = Flux
            if Flux == 0:  
                flux_zero += 1

    print('Max: ', max_count)
    print('Zero due to: \n- max travel: ', max_but_zero_count)
    print('- T_low:', zero_count)
    print('- flux:', flux_zero) 
    print('Flux: ', flux_count) 
    return f

def doer_of_thing(fix, m, num):
    """
    Gives bolometric L and R_ph (of evry observer)
    """
    rays_T, rays_den, rays, radii = ray_maker(fix, m, num)

    grad_E_tot = []
    idx_tot = []
    sphere_radius = []
    for i in range(len(rays_T)):
    # Calculate gradE for every ray
        temp = rays_T[i]
        dens = rays_den[i]
        ray = rays[i]
        _, _, photo = calc_photosphere(temp, dens, radii)
        sphere_radius.append(photo)
        grad_E, idx = grad_calculator(ray, radii, photo)
        grad_E_tot.append(grad_E)
        idx_tot.append(idx)

    # Calculate Flux and see how it looks
    flux = flux_calculator(grad_E_tot, idx_tot, 
                            rays, rays_T, rays_den)
    # plt.figure(figsize = [8,5])
    # plt.scatter(np.arange(192), flux, s = 3, color = 'orange')     
    # plt.xlabel('Observer')
    # plt.ylabel(r'Flux [erg/s cm$^2$]')
    # plt.grid()
    # plt.savefig('Figs/' + str(fix) + '_NEWflux.png')                   

    # Save flux
    with open('data/red/flux_m'+ str(m) + '_fix' + str(fix) + '.txt', 'a') as f:
        f.write('#snap '+ str(fix) + 'num ' + str(num) + ', ' + str(today) + '\n')
        f.write(' '.join(map(str, flux)) + '\n')
        f.close() 

    # Calculate luminosity 
    lum = np.zeros(len(flux))
    zero_count = 0
    neg_count = 0
    for i in range(len(flux)):
        # Turn to luminosity
        if flux[i] == 0:
            zero_count += 1
        if flux[i] < 0:
            neg_count += 1
            flux[i] = 0 
        lum[i] = flux[i] * 4 * np.pi * sphere_radius[i]**2

    # Average in observers
    lum = np.sum(lum)/192
    print('Tot zeros:', zero_count)
    print('Negative: ', neg_count)      
    print('Fix %i' %fix, ', Lum %.3e' %lum, '\n---------' )
    return lum, sphere_radius
#%%
##
# MAIN
##
if __name__ == "__main__":
    save = True
    plot = False
    m = 6 # Choose BH
    fixes, days, num_array = select_fix(m)
    lums = []
            
    for idx,fix in enumerate(fixes):
        lum, sphere_radius = doer_of_thing(fix, m, int(num_array[idx]))
        lums.append(lum)
    
    if save:
        if alice:
            pre = '/home/s3745597/data1/TDE/'
            np.savetxt('red_backup_save'+ str(m) + '.txt', (days, lums))
            np.savetxt(pre + 'tde_comparison/data/alicered'+ str(m) + '.txt', (days, lums))
        else:
            np.savetxt('data/red/new_reddata_m'+ str(m) + '.txt', (days, lums)) 
    #%% Plotting
    if plot:
        plt.figure()
        plt.plot(lums, '-o', color = 'maroon')
        plt.yscale('log')
        plt.ylabel('Bolometric Luminosity [erg/s]')
        plt.xlabel('Days')
        if m == 6:
            plt.title('FLD for $10^6 \quad M_\odot$')
            plt.ylim(1e41,1e45)
        if m == 4:
            plt.title('FLD for $10^4 \quad M_\odot$')
            plt.ylim(1e39,1e42)
        plt.grid()
        plt.savefig('Final plot/new_red' + str(m) + '.png')
        plt.show()

