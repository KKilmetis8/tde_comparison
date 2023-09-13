#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 16:02:14 2023

@author: konstantinos, paola 

Calculate the bolometric luminosity in the blued (BB) curve.
"""

import numpy as np
from emissivity import emissivity

# CHECK: t_eff = 1?
# question: which range of frequencies we integrate?

pi = 3.1416
c = 2.9979e10 #[cm/s]
h = 6.6261e-27 #[gcm^2/s]
Kb = 1.3806e-16 #[gcm^2/s^2K]
n_min = 4.8e14 #[Hz]
n_max = 1e15 #[Hz]
n_array = np.linspace(n_min,n_max, 1000)
loadpath = 'src/Optical_Depth/'

def planck_fun_n_cell(n,T):
    const = 2*h/c**2
    fun = const * n**3 / (np.exp(h*n/(Kb*T))-1)
    return fun

def planck_fun_cell(T):
    planck_fun_n_array = planck_fun_n_cell(n_array,T)
    fun = np.trapz(planck_fun_n_array, n_array)
    return fun

# QUESTION: we have T, rho from only one snapshoot? Does it not change?
def luminosity(n):
    lum = 0
    lnT_array = np.loadtxt(loadpath + 'T.txt') #it's in ln(CGS)
    lnrho_array = np.loadtxt(loadpath + 'rho.txt') #it's in ln(CGS)
    cell_vol_array = np.ones(len(lnT_array)) #we need an array of cell_vol 
    for i in range(0, len(lnT_array)):
        T = np.exp(lnT_array[i])
        rho = np.exp(lnrho_array[i])
        cell_vol = cell_vol_array[i]
        epsilon = emissivity(T, rho, cell_vol)
        lum_cell = epsilon * planck_fun_n_cell(n,T) / (planck_fun_cell(T) * np.exp(1))
        lum += lum_cell
    return 4*pi*lum

if __name__ == "__main__":
    test = luminosity()
    print(test)