#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18

@author: Paola

Gives the photosphere for red
"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

# Vanilla Imports
import numpy as np
import healpy as hp
from scipy.stats import gmean
import matplotlib.pyplot as plt
import matplotlib.colors as colors
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [5 , 3]
plt.rcParams['axes.facecolor'] = 'whitesmoke'

# Custom Imports
from src.Opacity.opacity_table import opacity

################
# FUNCTIONS
################
def get_kappa(T: float, rho: float, dr: float):
    '''
    Calculates the integrand of eq.(8) Steinberg&Stone22.
    CHECK THE IFs
    '''    
    # If there is nothing, the ray continues unimpeded
    if rho < np.exp(-49.3):
        return 0
    
    # Stream material, is opaque
    if T < np.exp(8.666):
        return 100
    
    # Too hot: Thompson Opacity.
    # Make it fall inside the table: from here the extrapolation is constant
    if T > np.exp(17.876):
        T = np.exp(17.87)
    
    # Lookup table
    k = opacity(T, rho,'red', ln = False)
    kappar =  k * dr
    
    return kappar

def calc_photosphere(T, rho, rs):
    '''
    Input: 1D arrays.
    Finds and saves the photosphere (in CGS).
    The kappas' arrays go from far to near the BH.
    '''
    threshold = 2/3
    kappa = 0
    kappas = []
    cumulative_kappas = []
    dr = rs[1]-rs[0] # Cell seperation
    i = -1 # Initialize reverse loop
    while kappa <= threshold and i > -len(T):
        new_kappa = get_kappa(T[i], rho[i], dr)
        kappa += new_kappa
        kappas.append(new_kappa)
        cumulative_kappas.append(kappa)
        i -= 1

    photo =  rs[i] #i it's negative
    return kappas, cumulative_kappas, photo

def get_photosphere(rays_T, rays_den, radii):
    '''
    Finds and saves the photosphere (in CGS) for every ray.

    Parameters
    ----------
    rays_T, rays_den: n-D arrays.
    radii: 1D array.

    Returns
    -------
    rays_kappas, rays_cumulative_kappas: nD arrays.
    photos: 1D array.
    '''
    # Get the thermalisation radius
    rays_kappas = []
    rays_cumulative_kappas = []
    rays_photo = np.zeros(len(rays_T))
    
    for i in range(len(rays_T)):
        # Isolate each ray
        T_of_single_ray = rays_T[i]
        Den_of_single_ray = rays_den[i]
        
        # Get photosphere
        kappas, cumulative_kappas, photo  = calc_photosphere(T_of_single_ray, Den_of_single_ray, radii)

        # Store
        rays_kappas.append(kappas)
        rays_cumulative_kappas.append(cumulative_kappas)
        rays_photo[i] = photo

    return rays_kappas, rays_cumulative_kappas, rays_photo

################
# MAIN
################

if __name__ == "__main__":
    from src.Calculators.ray_maker import ray_maker
    plot_kappas = False 
    plot_photosphere = True 
    m = 6 
    
    # Make Paths
    if m == 4:
        fixes = [233, 254, 263, 277 , 293, 308, 322]
        days = [1, 1.2, 1.3, 1.4, 1.56, 1.7, 1.8] 
        loadpath = '4/'
    if m == 6:
        fixes = [844, 881, 925, 950]
        days = [1, 1.1, 1.3, 1.4] #t/t_fb
        loadpath = '6/'

    fix_photo_arit = np.zeros(len(fixes))
    fix_photo_geom = np.zeros(len(fixes))
    for index, fix in enumerate(fixes):
        rays_T, rays_den, _, radii = ray_maker(fix, m)
        rays_kappa, rays_cumulative_kappas, rays_photo = get_photosphere(rays_T, rays_den, radii)
        rays_photo /=  6.957e10
        fix_photo_arit[index] = np.mean(rays_photo)
        fix_photo_geom[index] = gmean(rays_photo)
    
        if plot_kappas:
            plot_kappa = np.zeros( (len(radii), len(rays_kappa)))
            for i in range(192):
                for j in range(len(rays_cumulative_kappas)):
                    temp = rays_cumulative_kappas[i][j]
                    plot_kappa[-j-1,i] =  temp
                    if temp > 2/3:
                        print('Photosphere reached')
                        plot_kappa[0:-j, i ] = temp
                        break
                plot_kappa[0:-j, i ] = temp

            img = plt.pcolormesh(radii/6.957e10, np.arange(192), plot_kappa.T, 
                                cmap = 'Oranges', norm = colors.LogNorm(vmin = 1e-2, vmax =  2/3))
            cbar = plt.colorbar(img)
            plt.axvline(x=np.max(rays_photo), c = 'black', linestyle = '--')
            plt.title('Rays')
            cbar.set_label(r'$K_{ph}$')
            plt.xlabel('Distance from BH [$R_\odot$]')
            plt.ylabel('Observers')
            img.axes.get_yaxis().set_ticks([])
            plt.savefig('Final plot/photosphere.png')
            plt.show()

        print('Fix ', fix)

    if plot_photosphere:
        fix_photo_arit = "{:.4e}".format(fix_photo_arit)
        fix_photo_geom = "{:.4e}".format(fix_photo_geom)
        with open('data/photosphere_m' + str(m) + '.txt', 'a') as file:
                file.write('# t/t_fb \n')
                file.write(' '.join(map(str, days)) + '\n')
                file.write('# Photosphere arithmetic mean \n')
                file.write(' '.join(map(str, fix_photo_arit)) + '\n')
                file.write('# Photosphere geometric mean \n')
                file.write(' '.join(map(str, fix_photo_geom)) + '\n')
                file.close()
        plt.plot(days, fix_photo_arit, '-o', color = 'black', label = 'Photospehere radius, arithmetic mean')
        plt.plot(days, fix_photo_geom, '-o', color = 'pink', label = 'Photospehere radius, geometric mean')
        plt.xlabel(r't/$t_{fb}$')
        plt.ylabel(r'Photosphere [$R_\odot$]')
        plt.grid()
        plt.legend()
        plt.show()

        