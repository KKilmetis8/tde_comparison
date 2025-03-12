#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 12:07:22 2025

@author: konstantinos
"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

import numpy as np
import matplotlib.pyplot as plt
import colorcet

from src.Utilities.loaders import local_loader, alice_loader
from src.Utilities.parser import parse
import src.Utilities.prelude as c

from src.Utilities.isalice import isalice
alice, plot = isalice()

def masker(mask, list_of_quantities):
    new_list = []
    for quantity in list_of_quantities:
        new_list.append(quantity[mask])
    return (*new_list,)

#%%
alice, plot = isalice()
if alice:
    pre = '/home/kilmetisk/data1/TDE/'
else:
    pre = ''
 
if __name__ == '__main__':
    if alice:
        args = parse()
        if args.single:
            fixes = [args.only]
        else:
            fixes = np.arange(args.first, args.last + 1)
        save = True
        picset = 'normal'
        what = 'Density' # Density, Temperature, Dissipation
        m = int(np.log10(float(args.blackhole)))
    else:
        # Choose simulation
        m = 5
        fixes = [323]
        picset = 'normal'
        mstar = 0.5
        rstar = 0.47
        Rt = rstar * (10**m/mstar)**(1/3) 
        what = 'Dissipation' # Density, Temperature, Dissipation
        save = False
        plot = True
        args = None
        Mbh = 10**m
        Rt = rstar * (Mbh/mstar)**(1/3) 
        apocenter = Rt * (Mbh/mstar)**(1/3)

    for fix in fixes:
        if what == 'Temperature':
            load = 'midplane+T'
        if what == 'Density':
            load = 'midplane+Den'
        if what == 'Dissipation':
            load = 'midplane+Diss'
            
        X, Y, Z, Vol, Q, time = local_loader(m, fix, load)
        midplanemask = np.abs(Z) < 0.1 * Vol**(1/3)
        X, Y, Q, Vol = masker(midplanemask, [X, Y, Q, Vol])
        if what == 'Density':
            Q *= c.Msol_to_g / c.Rsol_to_cm**2
        if what == 'Dissipation':
            Q *= c.power_converter
        plot_Q = np.log10(Q)
        plot_Q = np.nan_to_num(plot_Q, neginf= 0)
        
        # if save:
        #     if alice:
        #         pre = f'/home/kilmetisk/data1/TDE/tde_comparison/data/denproj/paper'
        #         np.savetxt(f'{pre}/{m}{picset}{fix}.txt', den_plot)
        #         np.savetxt(f'{pre}/{m}{picset}x.txt', xs)
        #         np.savetxt(f'{pre}/{m}{picset}y.txt', ys)
        #     else:
        #         np.savetxt(f'data/denproj/paper/{m}{picset}{fix}.txt', den_plot) 
        #         # np.savetxt(f'data/xarray{m}.txt', xs) 
        #         # np.savetxt(f'data/yarray{m}.txt', ys) 
#%% Plot
        if plot:
            fig, ax = plt.subplots(1,1)
            ax.set_facecolor('k')
            plt.rcParams['figure.figsize'] = [8, 4]
            # Specify
            if what == 'Density':
                cb_text = r'Density [g/cm$^2$]'
                vmin = 0
                vmax = 5
            elif what == 'Temperature':
                cb_text = r'Temperature [K]'
                vmin = 3
                vmax = 7
            elif what == 'Dissipation':
                cb_text = r'Dissipated Energy [erg/s]'
                vmin = 30
                vmax = 40
            else:
                raise ValueError('Hate to break it to you champ \n \
                                but we don\'t have that quantity')
                    

            ax.set_xlabel(r' X $[R_\odot]$', fontsize = 14)
            ax.set_ylabel(r' Y $[R_\odot]$', fontsize = 14)
            step = 1
            img = ax.scatter(X[::step] / apocenter,# * c.Rsol_to_au, 
                             Y[::step] / apocenter,#* c.Rsol_to_au, 
                             c = plot_Q[::step], s = 1, 
                             alpha = 1, # np.abs(np.log10(Vol[::step])/np.max(np.log10(Vol))),
                             cmap = 'cet_fire',
                             vmin = vmin, vmax = vmax)
            cb = plt.colorbar(img)
            cb.set_label(cb_text, fontsize = 14)
            ax.set_xlim(-1.2, 0.2)
            ax.set_ylim(-0.2, 0.2)

            # ax.scatter(1,1, c = 'b')
            # ax.scatter(1,-1, c = 'b')
            # ax.scatter(-1,1, c = 'b')
            # ax.scatter(-1, 1, c = 'b')

            # time = np.loadtxt(f'{m}/{fixes[0]}/tbytfb_{fixes[0]}.txt')
            Mbh = 10**m
            tfb =  np.pi/np.sqrt(2) * np.sqrt( (rstar*c.Rsol_to_cm)**3/ (c.Gcgs*mstar*c.Msol_to_g) * Mbh/mstar)
            
            ax.set_title(f'10$^{m} M_\odot$ - {time*tfb/c.day_to_sec:.0f} days since disruption', #$t_\mathrm{{FB}}$',
                         fontsize = 16)
            