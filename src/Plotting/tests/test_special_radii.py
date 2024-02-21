"""
Created on January 2024
Author: Paola 
Midplane: observers 88-103

Check how behave Rph and Rth.

"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

import numpy as np
import matplotlib.pyplot as plt
import h5py
import src.Utilities.selectors as s
import src.Utilities.prelude as prel
from scipy.stats import gmean
from src.Calculators.ray_forest import ray_finder, ray_maker_forest
from src.Luminosity.special_radii_tree import get_specialr

m = 6
snap = 882
check = 'fid'
num = 1000
plot = 'spec_radii'
compare = True

opacity_kind = s.select_opacity(m)
filename = f"{m}/{snap}/snap_{snap}.h5"
Mbh = 10**m 
Rt =  Mbh**(1/3) # Msol = 1, Rsol = 1
apocenter = 2 * Rt * Mbh**(1/3)

thetas, phis, stops, xyz_grid = ray_finder(filename)
rays = ray_maker_forest(snap, m, check, thetas, phis, stops, num, opacity_kind)

if plot == 'spec_radii':
    # Plot Rph and Rth for all the observer in this snapshot
    _, _, rays_photo, _, _ = get_specialr(rays.T, rays.den, rays.radii, 
                                          rays.tree_indexes, opacity_kind, select = 'photo' )
    _, _, rays_thermr, _, _ = get_specialr(rays.T, rays.den, rays.radii, 
                                          rays.tree_indexes, opacity_kind, select = 'thermr_plot')
    rays_photo /= prel.Rsol_to_cm
    rays_thermr /= prel.Rsol_to_cm
    print(np.mean(rays_photo), np.mean(rays_thermr))
    print(gmean(rays_photo), gmean(rays_thermr))

    # fig, ax = plt.subplots(1,2, tight_layout = True)
    # # ax[0].scatter(np.arange(192), Elad_photo, c = 'k', s = 5, label = 'Elad')
    # ax[0].scatter(np.arange(192), rays_photo, c = 'orange', s = 5)
    # ax[0].set_xlabel('Observers')
    # ax[0].set_yscale('log')
    # ax[0].set_ylabel(r'$\log_{10}R_{ph} [R_\odot]$')
    # ax[0].grid()

    # #ax[1].scatter(np.arange(192), Elad_therm, c = 'k', s = 5, label = 'Elad')
    # ax[1].scatter(np.arange(192), rays_thermr, c = 'orange', s = 5)
    # ax[1].set_xlabel('Observers')
    # ax[1].set_yscale('log')
    # ax[1].set_ylabel(r'$\log_{10}R_{th} [R_\odot]$')
    # ax[1].grid()

    plt.scatter(np.arange(192), rays_photo, c = 'r', s = 8, label = r'$R_{ph}$')
    plt.scatter(np.arange(192), rays_thermr, c = 'b', s = 5, label = r'$R_{th}$')
    plt.xlabel('Observers')
    plt.yscale('log')
    plt.ylabel(r'R $[R_\odot]$')
    plt.axvline(88, linestyle = 'dashed', color = 'k', alpha = 0.4)
    plt.axvspan(88,103, color = 'aliceblue', alpha = 0.7)
    plt.axvline(103, linestyle = 'dashed', color = 'k', alpha = 0.4)
    plt.grid()
    plt.legend()
    #plt.ylim(0,8e3)
    # plt.title('Without mask from star')
    plt.savefig(f'Figs/specialR_m{m}_{snap}_origin0.png')

    if np.logical_and(m == 6, compare == True):
        with h5py.File(f'data/elad/data_{snap}.mat', 'r') as f:
            # print(f.keys())
            Elad_photo = f['r_photo'][0]
            Elad_therm = f['r_therm'][0]
        print(np.mean(Elad_photo), np.mean(Elad_therm))
        print(gmean(Elad_photo), gmean(Elad_therm))

        plt.figure()
        plot_ph = Elad_photo-rays_photo
        plto_th = Elad_therm-rays_thermr
        plt.plot(np.arange(192), plot_ph, c = 'r', label = r'$R_{ph}$')
        plt.plot(np.arange(192), plto_th, c = 'b', linestyle = 'dashed', label = r'$R_{th}$')
        plt.xlabel('Observers')
        plt.ylabel(r'$R^E-R^{us}$')
        #ax[0].set_yscale('log')
        plt.grid()
        plt.legend(fontsize = 10)
        plt.savefig(f'Figs/ABScomparison_special_radii{snap}.png')

    plt.show() 

if plot == 'profile':
    ## color with compton_cooling, T, rad_T, optical depth
    fig, ax = plt.subplots()
    selected_indexes = [90]
    for i in selected_indexes:
        radius = np.delete(rays.radii[i],-1)/prel.Rsol_to_cm
        img = ax.scatter(radius, rays.den[i], c = np.log10(rays.T[i]), s = 7, vmin = 5, vmax = 9)#label = f'observer {i}')
    cbar = fig.colorbar(img)
    cbar.set_label(r'$\log_{10}T [K]$')
    plt.loglog()
    ax.set_xlabel(r'$\log_{10}R [R_\odot]$')
    ax.set_ylabel(r'$\log_{10}\rho [g/cm^3]$')
    plt.xlim(10, 2*apocenter)
    #plt.legend()
    plt.savefig(f'Figs/profile{i}_{snap}.png')
   
    plt.show()
