#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 16:26:39 2023

@author: konstantinos

NOTES FOR OTHERS
- T, rho are in CGS
"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

import numpy as np
import matplotlib.pyplot as plt

# All units are cgs (NO log)
loadpath = 'src/Opacity/'
Tcool = np.loadtxt(loadpath + 'Tcool_ext.txt')
sig_abs = np.loadtxt(loadpath + 'sigma_abs.txt')

c = 2.99792458e10 #[cm/s]
alpha = 7.5646 * 10**(-15) # radiation density [erg/cm^3K^4]

def old_opacity(T, rho, kind) -> float:
    '''
    Return the rosseland mean opacity in [cgs], given a value of density,
    temperature and a kind of opacity. If ln = True, then T and rho are
    lnT and lnrho. Otherwise we convert them.
    
     Parameters
     ----------
     T : float,
         Temperature in [cgs].
     rho : float,
         Density in [cgs].
     kind : str,
         The kind of opacities. Valid choices are:
         rosseland, plank or effective.
     
    Returns
    -------
    opacity : float,
        The rosseland mean opacity in [cgs].
    '''    
    T = float(T)
    rho = float(rho)
    n = rho * 0.9 / (1.67e-24)

    interp_sig_abs = np.interp(T, Tcool, sig_abs)
    k_a = interp_sig_abs * n**2
    
    k_s = 0.34 * rho

    # Pick Opacity & Use Interpolation Function
    if kind == 'planck':
        kapparho = k_a
    
    elif kind == 'scattering':
        kapparho = k_s

    elif kind == 'effective':
        # STEINBERG & STONE (9) (Rybicky & Lightman eq. 1.98)
        kapparho = np.sqrt(3 * k_a * (k_a + k_s)) 
    
    elif kind == 'red':
        kapparho = k_a + k_s
        
    return kapparho

if __name__ == '__main__':
    from src.Opacity.opacity_table import opacity 
    import colorcet

    lnT = np.loadtxt(loadpath + 'T.txt')
    lnrho = np.loadtxt(loadpath + 'rho.txt')
    T = np.exp(lnT)
    rho = np.exp(lnrho)
    kappa_cloudy = np.zeros((len(T),len(rho)))
    kappa_lte = np.zeros((len(T),len(rho)))
    diff = np.zeros((len(T),len(rho)))
    logT = np.log10(T)
    logrho = np.log10(rho)

    for i in range(len(T)):
        for j in range(len(rho)):
            opacity_cloudy = old_opacity(T[i], rho[j], 'planck')
            kappa_cloudy[i][j] = np.log10(opacity_cloudy)
            # kappa_cloudy = np.nan_to_num(neginf=0)
            opacity_lte = opacity(T[i], rho[j], 'planck', ln = False)
            kappa_lte[i][j] = np.log10(opacity_lte)
            diff[i][j] = kappa_lte[i][j] - kappa_cloudy[i][j]

    fig, axs = plt.subplots(1,2, tight_layout = True)
    img = axs[0].pcolormesh(logrho, logT, kappa_cloudy, cmap = 'cet_rainbow', vmin = -20, vmax = 15)
    #cbar = plt.colorbar(img)
    # cbar.set_label(r'$\kappa^E$')
    axs[0].set_xlabel(r'$\log_{10}\rho [g/cm^3]$', fontsize = 15)
    axs[0].set_ylabel(r'$\log_{10}$T [K]', fontsize = 14)
    axs[0].title.set_text('CLOUDY')

    img1 = axs[1].pcolormesh(logrho, logT, kappa_lte, cmap = 'cet_rainbow', vmin = -20, vmax = 15)
    # cbar1 = plt.colorbar(img1)
    axs[1].set_xlabel(r'$\log_{10}\rho [g/cm^3]$', fontsize = 15)
    #axs[1].set_ylabel(r'$\log_{10}$T [K]', fontsize = 16)
    # cbar1.set_label(r'$\log_{10}\kappa [1/cm]$', fontsize = 15)
    axs[1].title.set_text('LTE')

    plt.suptitle(r'Opacity using $\rho$,T from tables')
    plt.savefig('Figs/opacitytables.png')

    plt.figure()
    img3 = plt.pcolormesh(logrho, logT, diff, cmap = 'cet_coolwarm', vmin = -6, vmax = 4)
    contours = plt.contour(logrho, logT, diff, levels  = 6,
                        colors = 'k', vmin = -6, vmax = 4)
    plt.clabel(contours, inline=True, fontsize = 12)
    plt.xlabel(r'$\log_{10}\rho [g/cm^3]$', fontsize = 15)
    plt.ylabel(r'$\log_{10}$T [K]', fontsize = 14)
    cbar3 = fig.colorbar(img3)
    cbar3.set_label(r'$\log_{10}(\kappa_{LTE}/\kappa_{CL})$', fontsize = 15)
    plt.title(r'Opacity using $\rho$,T from tables')
    plt.savefig('Figs/opacitytables_diff.png')
    plt.show()


