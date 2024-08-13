#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 13:52:21 2024

@author: konstantinos
"""
import numpy as np
# User lines
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

# Constants
c = 2.99792458e10 #[cm/s]
h = 6.62607015e-27 #[gcm^2/s]
Kb = 1.380649e-16 #[gcm^2/s^2K]
alpha = 7.5646 * 10**(-15) # radiation density [erg/cm^3K^4]
G = 6.6743e-11 # SI
sigma_T = 6.6524e-25 #[cm^2] thomson cross section
stefan = 5.67e-5 # [cgs]
me = 9.1e-28 # cgs
hbar =  6.662e-27 # # cgs
kb = 1.38e-16 # cgs
mp = 1.6749286e-24

mh2 = 2 * mp
mh = mp 
mhe = 4 * mp
# Solar units
Msol = 2e30 #1.98847e30 # kg
Rsol = 7e8 #6.957e8 # m
t = np.sqrt(Rsol**3 / (Msol*G )) # Follows from G = 1

# Ionization energies
ev_to_erg = 1.602e-12
xdis_h2 = 7.17e-12 # erg
xh = 13.598 * ev_to_erg # erg
prefactor_h = 1 # 2g1/g0
xhe1 = 24.587 * ev_to_erg # erg
prefactor_he1 = 4
xhe2 = 54.416 * ev_to_erg # erg
prefactor_he2 = 1
vib = 5984.48 # [K]
rot = 170.64 # [K]
 
# Converters
Rsol_to_cm = 6.957e10 # [cm]
Msol_to_g = 2e33 # 1.989e33 # [g]
den_converter = Msol_to_g / Rsol_to_cm**3
numden_converter = 1/Rsol_to_cm**3
en_den_converter = Msol_to_g / (Rsol_to_cm  * t**2 ) # Energy Density converter
sec_to_day = 60*60*24

# Healpy
NSIDE = 4

# Select opacity
def select_opacity(m):
    if m==6:
        return 'cloudy'
    else:
        return 'LTE'
    
# Plotting
import matplotlib.pyplot as plt
import colorcet
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [3, 3]
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.facecolor']= 'whitesmoke'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
AEK = '#F1C410'