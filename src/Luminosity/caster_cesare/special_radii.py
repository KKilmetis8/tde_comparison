#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 16:02:41 2023

@author: konstantinos, Paola

NOTES FOR OTHERS:
- things from snapshots are in solar and code units (mass in M_sol, 
  length in R_sol, time s.t. G=1), we have to convert them in CGS 

- change m, fixes, loadpath
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
from src.Luminosity.caster_cesare.ray_cesare import ray_maker

################
# FUNCTIONS
################
def spacing(t):
    start = 1
    end = 1.3
    n_start = 700
    n_end = 3000
    n = (t-start) * n_end - (t - end) * n_start
    n /= (end - start)
    return n

def select_fix(m):
    if m == 4:
        snapshots = [233] #, 254, 263, 277 , 293, 308, 322]
        days = [1]# , 1.2, 1.3, 1.4, 1.56, 1.7, 1.8] 
    if m == 6:
        snapshots = [844, 881, 925, 950, 1008] #[844, 881, 882, 898, 925, 950]
        days = [1, 1.1, 1.3, 1.4, 1.608] #[1, 1.139, 1.143, 1.2, 1.3, 1.4] # t/t_fb
            
        #     const = 0.05
    #     beginning = 1200
    # num_array = beginning * np.ones(len(snapshots))
    # for i in range(1,len(num_array)):
    #         num_array[i] = int(1.5 * num_array[i-1])
    num_array = [spacing(d) for d in days]
    return snapshots, days, num_array

def get_kappa(T: float, rho: float, dr: float):
    '''
    Calculates the integrand of eq.(8) Steinberg&Stone22.
    '''    
    # If there is nothing, the ray continues unimpeded
    if rho < np.exp(-49.3):
        return 0
    
    # Stream material, is opaque
    if T < np.exp(8.666):
        return 100
    
    # Too hot: Kramers' law for absorption (planck)
    if T > np.exp(17.876):
        X = 0.7389
        Z = 0.02
        kplanck =  1.2e26 * Z * (1 + X) * T**(-3.5) * rho #Kramers' bound-free opacity [cm^2/g]
        kplanck *= rho

        Tscatter = np.exp(17.87)
        kscattering = opacity(Tscatter, rho, 'scattering', ln = False)

        oppi = kplanck + kscattering
        tau_high = oppi * dr
        return tau_high 
    
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
    i = -1 # Initialize reverse loop
    while kappa <= threshold and i > -len(T):
        dr = rs[i]-rs[i-1] # Cell seperation
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


def optical_depth(T, rho, dr):
    '''
    Calculates the optical depth at a point

    Parameters
    ----------
    T : float,
        Temperature in [cgs]. 
    rho : float. 
        Density in [cgs]. 

    dr : float,
        Cell Size in R_sol.

    Returns
    -------
    tau : float,
        The optical depth in [cgs].
    '''    
    # If there is nothing, the ray continues unimpeded
    if rho < np.exp(-49.3):
        #print('rho small')
        return 0
    
    # Stream material, is opaque
    if T < np.exp(8.666):
        # print('T low')
        return 100
    
    # Too hot: Kramers' law
    if T > np.exp(17.876):
        # T = np.exp(17.87)
        X = 0.7389
        Z = 0.02
        kplanck =  1.2e26 * Z * (1 + X) * T**(-3.5) * rho #Kramers' bound-free opacity [cm^2/g]
        kplanck *= rho
        
        Tscatter = np.exp(17.87)
        kscattering = opacity(Tscatter, rho,'scattering', ln = False)
        oppi = np.sqrt(3 * kplanck * (kplanck + kscattering)) 
        tau_high = oppi * dr
        
        return tau_high 
    
    # Lookup table
    oppi = opacity(T, rho,'effective', ln = False)
    tau =  oppi * dr
    
    return tau

def calc_thermr(rs, T, rho, threshold = 1):
    '''
    Finds and saves the effective optical depth at every cell the ray passess through.
    We use it to find the thermr of the ray.

    Parameters
    ----------
    rs : arr
        Radial coordinates of a ray
    rho : arr,
        Densities in a ray.
    T : arr,
        Temperatures in a ray
    threshold : float, optional
        The desired optical depth. The default is 1.

    Returns
    -------
    taus : np.array,
        The optical depth of a single cell.
        
    thermr : float,
        Where the thermr is for that ray.

    cumulative_taus : np.array,
        The total optical depth of a single cell.
    '''
    tau = 0
    taus = []
    cumulative_taus = []
    i = -1 # Initialize reverse loop
    #print('--new ray--')
    while tau < threshold and i > -len(T):
        dr = rs[i]-rs[i-1] # Cell seperation
        new_tau = optical_depth(T[i], rho[i], dr)
        tau += new_tau
        taus.append(new_tau)
        cumulative_taus.append(tau)
        i -= 1
    thermr =  rs[i] #i it's negative
    return taus, thermr, cumulative_taus

def get_thermr(rays_T, rays_den, radii):
    # Get the thermr for every observer
    rays_tau = []
    rays_cumulative_taus = []
    thermr = np.zeros(len(rays_T))
    
    for i in range(len(rays_T)):
        
        # Isolate each ray
        T_of_single_ray = rays_T[i]
        Den_of_single_ray = rays_den[i]
        
        # Get thermr
        taus, th, cumulative_taus = calc_thermr(radii, T_of_single_ray, Den_of_single_ray, 
                                        threshold = 1)
        # Store
        rays_tau.append(taus)
        rays_cumulative_taus.append(cumulative_taus)
        thermr[i] = th

    return rays_tau, thermr, rays_cumulative_taus


################
# MAIN
################

if __name__ == "__main__":
    plot_tau_ph = False 
    plot_radii = False 
    m = 6 
    loadpath = str(m) + '/'

    snapshots, days, num_array = select_fix(m)
    fix_photo_arit = np.zeros(len(snapshots))
    fix_photo_geom = np.zeros(len(snapshots))
    
    fix_thermr_arit = np.zeros(len(snapshots))
    fix_thermr_geom = np.zeros(len(snapshots))

    for index in range(4,5):
        num = int(num_array[index])
        rays_T, rays_den, _, radii = ray_maker(snapshots[index], m, num)
        rays_kappa, rays_cumulative_kappas, rays_photo = get_photosphere(rays_T, rays_den, radii)
        rays_photo /=  6.957e10
        
        with open('data/red/photosphere' + str(snapshots[index]) + '_num' + str(num) + '.txt', 'a') as fileph:
            fileph.write(' '.join(map(str,rays_photo)) + '\n')
            fileph.close()

        fix_photo_arit[index] = np.mean(rays_photo)
        fix_photo_geom[index] = gmean(rays_photo)

        tau, thermr, cumulative_taus = get_thermr(rays_T, rays_den, radii)
        thermr /=  6.957e10
        fix_thermr_arit[index] = np.mean(thermr)
        fix_thermr_geom[index] = gmean(thermr)
    
        if plot_tau_ph:
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
            plt.axvline(x=np.mean(rays_photo), c = 'k', linestyle = '--', label = r'$\bar{R}_{ph}$ arit mean')
            plt.axvline(x=gmean(rays_photo), c = 'b', linestyle = '--', label = r'$\bar{R}_{ph}$ geom mean')
            plt.title('Rays')
            cbar.set_label(r'$\tau_{ph}$')
            plt.xlabel(r'Distance from BH [$\log_{10}R_\odot$]')
            plt.ylabel('Observers')
            img.axes.get_yaxis().set_ticks([])
            plt.legend()
            plt.xscale('log')
            plt.savefig('Final plot/photosphere_' + str(snapshots[index]) + 'num_' + str(int(num_array[index])) + '.png')
            plt.show()

        print('Fix', snapshots[index])
        
    if plot_radii:
        with open('data/special_radii_m' + str(m) + '.txt', 'a') as file:
                file.write('# t/t_fb \n')
                file.write(' '.join(map(str, days)) + '\n')
                file.write('# Photosphere arithmetic mean \n')
                file.write(' '.join(map(str, fix_photo_arit)) + '\n')
                file.write('# Photosphere geometric mean \n')
                file.write(' '.join(map(str, fix_photo_geom)) + '\n')
                file.write('# Thermalisation radius arithmetic mean \n')
                file.write(' '.join(map(str, fix_thermr_arit)) + '\n')
                file.write('# Thermalisation radius geometric mean \n')
                file.write(' '.join(map(str, fix_thermr_geom)) + '\n')
                file.close()
        plt.plot(days, fix_photo_arit, '-o', color = 'black', label = 'Photospehere radius, arithmetic mean')
        plt.plot(days, fix_photo_geom, '-o', color = 'pink', label = 'Photospehere radius, geometric mean')
        plt.plot(days, fix_thermr_arit, '-o', color = 'b', label = 'Thermalization radius, arithmetic mean')
        plt.plot(days, fix_thermr_geom, '-o', color = 'r', label = 'Thermalization radius, geometric mean')
        plt.xlabel(r't/$t_{fb}$')
        plt.ylabel(r'Photosphere [$R_\odot$]')
        plt.grid()
        plt.legend()
    plt.show()

        