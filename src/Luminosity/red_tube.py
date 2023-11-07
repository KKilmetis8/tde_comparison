#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 17:55:24 2023

@author: konstantinos
"""
import numpy as np
import numba
from src.Opacity.opacity_table import opacity
from src.Calculators.raymaker_tube import ray_maker, isalice
alice = isalice()
from src.Luminosity.special_radii_tube import get_photosphere

# Setup
if alice:
    fixes = np.arange(845, 1005, step = 10)
else:
    fixes = [844]
m = 6
# Constants
c_cgs = 3e10 # [cm/s]
Rsol_to_cm = 6.957e10 # [cm]
pre = '/home/s3745597/data1/TDE/'
#%%
@numba.njit
def grad_calculator(ray: np.array, radii: np.array, r_photo): 
    # For a single ray (in logspace) get 
    # the index of radius closest to sphere_radius and the gradE there.
    # Everything is in CGS.
    for i, radius in enumerate(radii):
        if radius > r_photo:
            idx = i - 1 
            break
    
    # For rad
    if idx<0:
        print('Bad Observer, photosphere is the closest point')
        idx=0
        
    step = radii[idx+1] - radii[idx]
    grad_E = (ray[idx+1] - ray[idx]) / step 

    return grad_E, idx

    
def flux_calculator(grad_E, idx, 
                    single_Rad, single_T, single_Den):
    """
    Get the flux for every observer.
    Eevrything is in CGS.

    Parameters: 
    grad_E idx_tot are 1D-array of lenght = len(rays)
    rays, rays_T, rays_den are len(rays) x N_cells arrays
    """
    # We compute stuff OUTSIDE the photosphere
    # (which is at index idx_tot[i])
    idx = idx+1 #  
    Energy = single_Rad[idx]
    max_travel = np.sign(-grad_E) * c_cgs * Energy # or should we keep the abs???
    
    Temperature = single_T[idx]
    Density = single_Den[idx]
    
    # Ensure we can interpolate
    rho_low = np.exp(-45)
    T_low = np.exp(8.77)
    T_high = np.exp(17.8)
    
    # If here is nothing, light continues
    if Density < rho_low:
        return max_travel
    
    # If stream, no light 
    if Temperature < T_low: 
        return 0
    
    # T too high => Kramers'law
    if Temperature > T_high:
        X = 0.7389
        k_ross = 3.68 * 1e22 * (1 + X) * Temperature**(-3.5) * Density #Kramers' opacity [cm^2/g]
        k_ross *= Density
    else:    
        # Get Opacity, NOTE: Breaks Numba
        k_ross = opacity(Temperature, Density, 'rosseland', ln = False)
    
    # Calc R, eq. 28
    R = np.abs(grad_E) /  (k_ross * Energy)
    invR = 1 / R
    # Calc lambda, eq. 27
    coth = 1 / np.tanh(R)
    lamda = invR * (coth - invR)
    # Calc Flux, eq. 26
    Flux = - c_cgs * grad_E  * lamda / k_ross
    
    # Take the minimum between F, cE
    if Flux > max_travel:
        return max_travel
    else:
        return Flux

def red(fixes, m):

    bols = []
    photos = []
    for d, fix in enumerate(fixes):
        rays_T, rays_Den, rays_Rad, rays_R = ray_maker(fix, m)
        cumul_kappa, photo =  get_photosphere(rays_T, rays_Den, rays_R)
        fluxes = []
        for i in range(len(rays_T)):
        
            # Isolate ray
            single_T = rays_T[i]
            single_Den = rays_Den[i]
            single_Rad = rays_Rad[i]
            single_radii = rays_R[i]
            
            #  Calculate gradE
            grad_E, idx = grad_calculator(single_Rad, single_radii, photo[i])
        
            # Calculate Flux 
            flux = flux_calculator(grad_E, idx, 
                                    single_Rad, single_T, single_Den)
            fluxes.append(flux) # Keep it
            
        # Convert to luminosity 
        lum = np.zeros(len(fluxes))
        for i in range(len(fluxes)):
            # Ignore negative flux
            if fluxes[i] < 0:
                continue
            # Convert photo to cm
            r = photo[i] * Rsol_to_cm
            lum[i] = fluxes[i] * 4 * np.pi * r**2
        
        # Average in observers
        bol_lum = np.sum(lum)/192
        print('L FLD: %.2e' % bol_lum)
        # Hold
        bols.append(bol_lum)
        photos.append(photo)
        
    if alice:
        np.savetxt(pre + 'data/red_tube_alice', bols)
        np.savetxt(pre + 'data/photosphere_alice', photos)
    else:
        np.savetxt('data/red_tube',bols)
        np.savetxt('data/photosphere', bols)
        
    if not alice:
        from src.Utilities.finished import finished
        finished()
            
    return bols, photos
#%% Main

bols, photos = red(fixes, m)
#%% Plot & Print

plot = True 
if plot and not alice:
    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = True
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['figure.figsize'] = [5 , 4]
    plt.rcParams['axes.facecolor']= 	'whitesmoke'
    plt.rcParams['figure.figsize'] = [5 , 4]
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    AEK = '#F1C410' # Important color 
    # Choose fix
    which = 0
    lum = bols[which]
    photo = photos[which]
    
    # Plot
    fig, ax = plt.subplots(1,2, figsize = (10, 6))
    ax[0].plot(lum, 'o', c ='k')
    
    # Make pretty
    ax[0].set_yscale('log')
    ax[0].set_title('FLD Luminosity')
    ax[0].set_xlabel('Observers')
    ax[0].set_ylabel('Luminosity [erg/s]')
    
    # Plot Photo
    ax[1].plot(photo, 'o', c = 'k')
    
    ## Make pretty
    # My lines
    from scipy.stats import gmean
    ax[1].axhline(np.mean(photo), linestyle = '--' ,c = 'seagreen')
    ax[1].axhline(gmean(photo), linestyle = '--', c = 'darkorange')
    #ax[1].text()
    # Elad's lines
    ax[1].axhline(40 , c = AEK)
    ax[1].axhline(700, c= 'lightseagreen')
    
    ax[1].set_yscale('log')
    ax[1].set_title('Photosphere')
    ax[1].set_ylabel(r'Photosphere Radius [$R_\odot$]')
    ax[1].set_xlabel('Observers')
    
