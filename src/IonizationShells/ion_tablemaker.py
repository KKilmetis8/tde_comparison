#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 12:49:17 2024

@author: konstantinos
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect, brentq, fsolve
from tqdm import tqdm

import src.Utilities.prelude as c
#%%
Ts = np.logspace(3, 11, num=100)
Dens = np.logspace(-13, 2, num = 100)
H_table = np.zeros((100, 100))
He1_table = np.zeros((100,100))
He2_table = np.zeros((100,100)) 
Vol = 1e30

#%%
class partition:
    # Straight out of Tomida
    def __init__(self, T, V):
        # Molecular Hydrogen
        Ztr_H2 = (2 * np.pi * c.mh2 * c.kb * T)**(3/2) / c.h**3
        Zrot_even_H2 = 0
        Zrot_odd_H2 = 0
        for i in range(0, 15, 2):
            odd = i+1
            even = i
            Zrot_even_H2 += (2*even+1) * np.exp(- even *
                                                (even + 1) * c.rot / (2*T))
            Zrot_odd_H2 += (2*odd+1) * np.exp(- odd *
                                              (odd + 1) * c.rot / (2*T))
        Zrot = Zrot_even_H2**(1/4) * (3*Zrot_odd_H2 * np.exp(c.rot/T))**(3/4)
        Zvib_H2 = 1 / (2 * np.sinh(c.vib / (2*T)))  # CHANGE THIS
        #Zvib_H2 = 1 / (1 -  np.exp(c.vib/T))
        Zspin_H2 = 4
        Zelec_H2 = 2
        self.Z_H2 = V * Ztr_H2 * Zrot * Zvib_H2 * Zspin_H2 * Zelec_H2

        # Atomic Hydrogen
        Ztr_H = (2 * np.pi * c.mh * c.kb * T)**(3/2) / c.h**3
        Zspin_H = 2
        Zelec_H = 2 * np.exp(- c.xdis_h2 / (2 * c.kb * T))
        self.Z_H = V * Ztr_H * Zspin_H * Zelec_H

        # Ionized Hydrogen
        Ztr_Hion = (2 * np.pi * c.mh * c.kb * T)**(3/2) / c.h**3
        Zspin_Hion = 2  # NOTE CHANGED FROM 2
        Zelec_Hion = 2 * np.exp(- (c.xdis_h2 + 2*c.xh) / (2 * c.kb * T))
        self.Z_Hion = V * Ztr_Hion * Zspin_Hion * Zelec_Hion

        # Atomic Helium
        Ztr_He = (2 * np.pi * c.mhe * c.kb * T)**(3/2) / c.h**3
        self.Z_He = V * Ztr_He

        # 1 Ionized Helium
        # Ztr_He1 = 2 * np.pi * c.mhe * c.kb * T**(3/2) / c.h**3
        Zelec_He1 = np.exp(- c.xhe1 / (c.kb * T))
        self.Z_He1 = V * Ztr_He * Zelec_He1

        # 2 Ionized Helium
        Zelec_He2 = np.exp(- (c.xhe1+c.xhe2) / (c.kb * T))
        self.Z_He2 = V * Ztr_He * Zelec_He2

        # Electron
        Ztr_e = (2 * np.pi * c.me * c.kb * T)**(3/2) / c.h**3
        Zspin_e = 2
        self.Z_e = V * Ztr_e * Zspin_e

def bisection(f, a, b, *args,
              tolerance=1e-1,
              max_iter=int(1e2)):
    rf_step = 0
    # Sanity
    fa = f(a, *args)
    fb = f(b, *args)
    if fa * fb > 0:
        print('No roots here bucko')
        return a
    while rf_step < max_iter:
        rf_step += 1
        c = 0.5 * (a + b)
        fc = f(c, *args)
        if np.abs(fc) < tolerance:
            break
        if np.sign(fc) == np.sign(fa):
            a = c
        else:
            b = c
    return c

def chemical_eq(ne, i):
    oros1 = 2 * ne**2 * nH * K_ion / \
        (np.sqrt((ne + K_ion)**2 + 8 * nH
         * ne**2 / K_dis) + ne + K_ion)
    oros2 = (K_He1 * ne + 2 * K_He1 * K_He2) * nHe * \
        ne**2 / (ne**2 + K_He1 * ne + K_He1 * K_He2)
    oros3 = -ne**3
    return oros1 + oros2 + oros3

for i, T in enumerate(Ts):
    for j, Den in enumerate(Dens):
        par = partition(T, Vol)
        K_dis = par.Z_H**2 / (par.Z_H2 * Vol)
        K_ion = par.Z_Hion * par.Z_e / (par.Z_H * Vol)
        K_He1 = par.Z_He1 * par.Z_e / (par.Z_He * Vol)
        K_He2 = par.Z_He2 * par.Z_e / (par.Z_He1 * Vol)
        
        # Abundances
        X = 0.7
        Y = 0.28
        nH = Den * X / c.mh
        nHe = Den * Y / c.mhe
        ne_sol = bisection(chemical_eq, 1e-15, 1e50, i)
        inv_ne_sol = 1/ne_sol

        #--- Get Ion fraction
        # He
        orosHe = 1 + K_He1 * inv_ne_sol + K_He1 * K_He2*inv_ne_sol**2
        nHe_sol = np.divide(nHe, orosHe)  # Eq 83
        
        # He+
        n_He1_sol = nHe_sol * K_He1 * inv_ne_sol  # Eq 77
        xHe1 = n_He1_sol / (n_He1_sol + nHe_sol)
        
        # He++
        n_He2_sol = nHe_sol * K_He2 * inv_ne_sol  # Eq 78
        xHe2 = n_He2_sol / (n_He2_sol + nHe_sol)
        
        # H
        orosH = ne_sol - nHe_sol * inv_ne_sol * K_He1 - \
            2 * nHe_sol - inv_ne_sol**2 * K_He1 * K_He2
        alpha = 2 / K_dis
        beta = 1 + K_ion/ne_sol
        gamma = -nH
        delta = beta**2 - 4 * alpha * gamma
        nH_sol = (-beta + np.sqrt(delta)) / (2*alpha)
        # nH_sol2 = (-beta - np.sqrt(delta)) / (2*alpha)
        #nH_sol = np.divide(ne_sol, K_ion) * orosH  # Eq 84
        
        # H+
        n_Hion_sol = nH_sol * K_ion * inv_ne_sol  # Eq 76
        xH = n_Hion_sol / (n_Hion_sol + nH_sol)
    
        #--- Store
        H_table[i][j] = xH
        He1_table[i][j] = xHe1
        He2_table[i][j] = xHe2

H_table = np.nan_to_num(H_table, nan = 1)
np.save('src/IonizationShells/xH', H_table)
np.save('src/IonizationShells/xHe1', He1_table)
np.save("src/IonizationShells/xHe2", He2_table)
