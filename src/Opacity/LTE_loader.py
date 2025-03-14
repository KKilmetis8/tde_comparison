#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 18:55:57 2024

@author: konstantinos

Loads the opacity data and extrapolates
Quantities X remain in lnX form.
"""

import numpy as np
from src.Opacity.linextrapolator import pad_interp, extrapolator_flipper, nouveau_rich

opac_kind = 'LTE'
opac_path = f'src/Opacity/{opac_kind}_data/'
T_cool = np.loadtxt(f'{opac_path}/T.txt')
Rho_cool = np.loadtxt(f'{opac_path}/rho.txt')
plank = np.loadtxt(f'{opac_path}/planck.txt')
rossland = np.loadtxt(f'{opac_path}/ross.txt')
scattering = np.loadtxt(f'{opac_path}/scatter.txt')



_, _, plank_ex = nouveau_rich(T_cool, Rho_cool, plank, what = 'abs', slope_length=5)
T_opac_ex, Rho_opac_ex, rossland_ex = nouveau_rich(T_cool, Rho_cool, rossland, what = 'scattering',
                                                   slope_length = 5)
_, _, rossland_ex2 = nouveau_rich(T_cool, Rho_cool, rossland, what = 'abs')

_, _, scattering_ex = nouveau_rich(T_cool, Rho_cool, scattering, what = 'scattering')
_, _, scattering_ex2 = nouveau_rich(T_cool, Rho_cool, scattering, what = 'abs')


# Test the -3.5 thing
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import colorcet
    plt.figure()
    tplot0 = np.log10(np.exp(T_cool))
    dplot0 = np.log10(np.exp(Rho_cool))
    splot0 = np.log10(np.exp(plank))# / np.exp(Rho_cool))
    cb = plt.pcolormesh(tplot0, dplot0, splot0.T, 
                        cmap = 'cet_CET_CBL2_r', vmin = -7, vmax = 0)
    plt.figure()
    tplot = np.log10(np.exp(T_opac_ex))
    dplot = np.log10(np.exp(Rho_opac_ex))
    splot = np.log10(np.exp(plank_ex))#  / np.exp(Rho_opac_ex))
    # splot = np.log10(np.subtract( np.exp(scattering_ex), np.exp(scattering_ex2)))

    cb = plt.pcolormesh(tplot, dplot, splot.T, 
                        cmap = 'cet_CET_CBL2_r', vmin = -15, vmax = -4)
    plt.colorbar(cb)
    plt.xlabel(r'Temperature, $\log_\mathrm{10} T$ [K]')
    plt.ylabel(r'Density, $\log_\mathrm{10} \rho $ [g/cm$^3]$')
    
    plt.axvline(np.log10(np.exp(T_cool))[0], c = 'r', ls = '--', alpha = 0.1)
    plt.axvline(np.log10(np.exp(T_cool))[-1], c = 'r', ls = '--')
    plt.axhline(np.log10(np.exp(Rho_cool))[0], c = 'r', ls = '--')
    plt.axhline(np.log10(np.exp(Rho_cool))[-1], c = 'r', ls = '--')
    
    
