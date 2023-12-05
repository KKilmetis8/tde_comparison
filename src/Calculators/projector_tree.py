#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project quantities.

@author: paola

"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

import numpy as np
import matplotlib.pyplot as plt
from src.Calculators.ray_forest import ray_maker
alice = False

#%% Constants & Converter
NSIDE = 4
G = 6.6743e-11 # SI
Msol = 1.98847e30 # kg
Rsol = 6.957e8 # m
t = np.sqrt(Rsol**3 / (Msol*G )) # Follows from G = 1
c = 3e8 * t/Rsol # simulator units. Need these for the PW potential
c_cgs = 3e10 # [cm/s]
Msol_to_g = 1.989e33 # [g]
Rsol_to_cm = 6.957e10 # [cm]
den_converter = Msol_to_g / Rsol_to_cm**2
en_den_converter = Msol_to_g / (Rsol_to_cm  * t**2 ) # Energy Density converter

fix = 844
m = 6
num = 1000

def isalice():
    return alice

def projector(fix, m):
    # Make the 3D grid
    _, gridded_den, gridded_mass, x_radii, y_radii, z_radii = ray_maker(fix,m)
    flat_den =  np.zeros(( len(x_radii), len(y_radii) ))
    # flat_mass =  np.zeros(( len(x_radii), len(y_radii) ))
    for i in range(len(x_radii)):
        for j in range(len(y_radii)):
                mass_zsum = 0
                for k in range(len(z_radii)):
                    mass_zsum += gridded_mass[i,j,k]
                    flat_den[i,j] += gridded_den[i,j,k] * gridded_mass[i,j,k]
                flat_den[i,j] = np.divide(flat_den[i,j], mass_zsum)
    return flat_den, x_radii, y_radii
 
if __name__ == '__main__':
    m = 6
    num = 1000
    flat_den, x_radii, y_radii = projector(844, m)

#%% Plot
    plot = True
    if plot:
        import colorcet
        fig, ax = plt.subplots(1,1)
        plt.rcParams['text.usetex'] = True
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['figure.figsize'] = [6, 4]
        plt.rcParams['axes.facecolor']= 	'whitesmoke'
        
        den_plot = np.log10(flat_den)
        den_plot = np.nan_to_num(den_plot, neginf= -19)
  
        ax.set_xlim(-15_000, 500)
        ax.set_ylim(-5_000, 8000)
        ax.set_xlabel(r' X [$_\odot$]', fontsize = 14)
        ax.set_xlabel(r' Y [R$_\odot$]', fontsize = 14)
        img = ax.pcolormesh(x_radii, y_radii, den_plot.T, cmap = 'cet_fire',
                            vmin = 0, vmax = 5)
        cb = plt.colorbar(img)
        cb.set_label(r'Density [g/cm$^2$]', fontsize = 14)
        ax.set_title('Midplane', fontsize = 16)
        plt.show()
    