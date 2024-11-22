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
Lsol_to_ergs = 3.846e33
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
Gcgs = 6.6743e-8 # cgs
den_converter = Msol_to_g / Rsol_to_cm**3
numden_converter = 1/Rsol_to_cm**3
en_den_converter = Msol_to_g / (Rsol_to_cm  * t**2 ) # Energy Density converter
day_to_sec = 60*60*24
sec_to_yr = 1 / (60*60*24*365)

# Healpy
import healpy as hp
NSIDE = 4
NPIX = hp.nside2npix(NSIDE )#  int(NSIDE * 96)

# Select opacity
def select_opacity(m):
    if m==6:
        return 'cloudy'
    else:
        return 'LTE'
    
# Frame of ref
def set_change(m):
    if m == 4:
        change = 80
    if m == 5:
        change = 132
    if m == 6:
        change = 180
    return change

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

# 6 palette
darkb = '#264653'
cyan = '#2A9D8F'
prasinaki = '#6a994e'
yellow = '#E9C46A'
kroki = '#F2A261'
reddish = '#E76F51'


# 9 palette
c91 = '#e03524'
c92 = '#f07c12'
c93 = '#ffc200'
c94 = '#90bc1a'
c95 = '#21b534'
c96 = '#0095ac'
c97 = '#1f64ad'
c98 = '#4040a0'
c99 = '#903498'

r20_palette = [
    "#D32F2F", "#E64A19", "#F57C00", "#FFA000", "#FBC02D",  # Reds to Oranges
    "#FDD835", "#C0CA33", "#7CB342", "#388E3C", "#00897B",  # Yellows to Greens
    "#0097A7", "#0288D1", "#1976D2", "#303F9F", "#512DA8",  # Blues to Indigos
    "#673AB7", "#7B1FA2", "#6A1B9A", "#4A148C", "#311B92"   # Purples
]
r10_palette = [
    "#D32F2F", "#F57C00", "#FBC02D", "#7CB342", "#388E3C",  # Reds to Greens
    "#0097A7", "#1976D2", "#512DA8", "#7B1FA2", "#4A148C"   # Blues to Purples
]

