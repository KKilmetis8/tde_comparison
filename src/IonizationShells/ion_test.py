#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 18:24:15 2024

@author: konstantinos
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect, brentq, fsolve
from tqdm import tqdm

import src.Utilities.prelude as c
# %%
T = np.logspace(3.5, 4.8, num=100)
# T = np.array([4000])
Vol = np.ones_like(T) * 1e30
Den = np.ones_like(T) * 1e-10 #1.8e2

# %%


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


par = partition(T, Vol)
K_dis = par.Z_H**2 / (par.Z_H2 * Vol)
K_ion = par.Z_Hion * par.Z_e / (par.Z_H * Vol)
# logK_ion = np.log10(K_ion)
K_He1 = par.Z_He1 * par.Z_e / (par.Z_He * Vol)
K_He2 = par.Z_He2 * par.Z_e / (par.Z_He1 * Vol)
# logK_He2 = np.log10(K_He2)

# Abundances
X = 1  # 0.7
Y = 0  # 0.28# 0.28
nH = Den * X / c.mh
print(nH[0] * 1e-6, 'in m3')
nHe = Den * Y / c.mhe
# %%


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
        print('f(a): %.1e' % fa)
        print('f(b): %.1e' % fb)
        print('f(c): %.1e' % fc)
        if np.abs(fc) < tolerance:
            break
        if np.sign(fc) == np.sign(fa):
            a = c
        else:
            b = c
    print('B steps: ', rf_step)
    return c


def chemical_eq(ne, i):
    oros1 = 2 * ne**2 * nH[i] * K_ion[i] / \
        (np.sqrt((ne + K_ion[i])**2 + 8 * nH[i]
         * ne**2 / K_dis[i]) + ne + K_ion[i])
    oros2 = (K_He1[i] * ne + 2 * K_He1[i] * K_He2[i]) * nHe[i] * \
        ne**2 / (ne**2 + K_He1[i] * ne + K_He1[i] * K_He2[i])
    oros3 = -ne**3
    return oros1 + oros2 + oros3


ne_sol = np.zeros(len(nH))
for i in range(len(nH)):
    # ne_sol[i] = bisect(chemical_eq, 1e-10, 1e20, args = i, disp = False)
    #ne_sol[i] = brentq(chemical_eq, 1e-1, 1e21, args = (i), maxiter=1000, disp = False)
    ne_sol[i] = bisection(chemical_eq, 1e-15, 1e34, i)
inv_ne_sol = 1/ne_sol
# plt.scatter(T, chemical_eq(ne_sol, np.arange(len(T))))

# %%
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
#%% Literature
def saha(x, i):
    maxwell = (2 * np.pi * c.me * c.kb * T[i])**(3/2) / c.h**3 
    expon = np.exp( - c.xh / (c.kb * T[i]))
    res =  x**2 - (1-x) * 1/nH[i] * maxwell * expon
    return res

def saha_anal(i):
    maxwell = (2 * np.pi * c.me * c.kb * T[i])**(3/2) / c.h**3 
    expon = np.exp( - c.xh / (c.kb * T[i]))
    alpha = 1
    beta = 1/nH[i] * maxwell * expon
    gamma = - beta
    delta = beta**2 - 4 * alpha * gamma
    sol1 = (-beta + np.sqrt(delta)) / (2*alpha)
    sol2 = (-beta - np.sqrt(delta)) / (2*alpha)
    if sol1>0:
        return sol1
    else:
        return sol2
saha_sol = np.zeros(len(nH))
saha_anal_sol = np.zeros(len(nH))
for i in range(len(nH)):
    saha_sol[i] = bisection(saha, 0, 1, i, max_iter = 100, tolerance = 1e-3)
    saha_anal_sol[i] = saha_anal(i)
# %%
plt.figure(figsize=(4, 4))
plt.grid()
plt.plot(T, xH, c='k', ls =':',lw=2, label='Tomida')
# plt.plot(T, xHe1, c = c.AEK, lw = 2)
# plt.plot(T, xHe2, c='maroon', lw = 2)
plt.ylim(-0.1, 1.1)
plt.xscale('log')

# Mathematica
numden = nH[0] * 1e6
if numden < 1e10:
    mathxH = [0.000894976, 0.0141422, 0.135873, 0.63172, 0.96862, 0.997991, \
0.999806, 0.999973, 0.999995, 0.999999, 1., 1., 1., 1., 1., 1.]
    mathT = [5000., 6000., 7200., 8640., 10368., 12441.6, 14929.9, 17915.9, \
21499.1, 25798.9, 30958.7, 37150.4, 44580.5, 53496.6, 64195.9, 77035.1]
else:
    mathxH = [0.0000430592, 0.000149286, 0.000429364, 0.00105542, 0.0022717,
              0.00436669, 0.0076191, 0.0122384, 0.0183302, 0.0259032, 0.03491,
              0.0452926, 0.0570135, 0.0700658, 0.0844651, 0.100235, 0.117389,
              0.135924, 0.155816, 0.177031, 0.199535, 0.223303, 0.248329, 0.274619,
              0.302186, 0.331037, 0.361161, 0.392518, 0.425028, 0.45856, 0.49293,
              0.527898, 0.563179, 0.598449, 0.633376, 0.667635, 0.700936, 0.733039,
              0.763756, 0.79295, 0.820512, 0.846335, 0.870281, 0.892167, 0.911771,
              0.928901, 0.943469, 0.955545, 0.965339, 0.973147, 0.979294]
    mathT = [10000., 12000., 14400., 17280., 20736., 24883.2, 29859.8, 35831.8,
             42998.2, 51597.8, 61917.4, 74300.8, 89161., 106993., 128392.,
             154070., 184884., 221861., 266233., 319480., 383376., 460051.,
             552061., 662474., 794968., 953962., 1.14475*1e6, 1.37371*1e6,
             1.64845*1e6, 1.97814*1e6, 2.37376*1e6, 2.84852*1e6, 3.41822*1e6,
             4.10186*1e6, 4.92224*1e6, 5.90668*1e6, 7.08802*1e6, 8.50562*1e6,
             1.02067*1e7, 1.22481*1e7, 1.46977*1e7, 1.76373*1e7, 2.11647*1e7,
             2.53977*1e7, 3.04772*1e7, 3.65726*1e7, 4.38871*1e7, 5.26646*1e7,
             6.31975*1e7, 7.5837*1e7, 9.10044*1e7]

# plt.plot(mathT, mathxH, c='r', ls='--', label='Mathematica')

# Literature
x = [7435.632655644819, 7865.01930064692, 8449.9550768907, 8965.83650028168, 8744.891585503297, 8190.471785803063, 6563.507717198775, 5757.646246849538, 5243.346200431279, 8854.674932955562, 8636.4693702743, 9021.939514872178, 9221.079114817276, 9307.76463197599,
     9424.614286269023, 9632.642051863178, 9753.570216486289, 9968.858965401718, 10284.683547962955, 10511.695463075788, 11223.240810591691, 12285.707678704597, 13117.337378254962, 14269.819795189262, 15669.49258094625, 16834.86121588754, 20682.818370995574]
y = [0.0061728395061730335, 0.04526748971193423, 0.1872427983539095, 0.39917695473251036, 0.2962962962962964, 0.10288065843621419, -0.008230452674897082, -0.014403292181069838, -0.014403292181069838, 0.3436213991769548, 0.24691358024691362, 0.44650205761316875, 0.5123456790123457,
     0.5699588477366256, 0.6419753086419753, 0.7037037037037037, 0.7469135802469137, 0.8106995884773662, 0.8806584362139918, 0.9382716049382717, 0.9732510288065843, 0.9979423868312758, 1.0061728395061729, 1.0020576131687244, 1.0020576131687244, 1.0020576131687244, 1]
x = np.array(x)
sorter = np.argsort(x)
y = np.array(y)
x = x[sorter]
y = y[sorter]
# plt.plot(x, y, c='b', label='Literature')
plt.plot(T, saha_anal_sol, c='g', label='Analytic Saha', lw = 3)
plt.plot(T, saha_sol, c='gold', ls = '--', label = 'Numerical Saha' )

# Pretty
plt.legend(fontsize=8, ncol = 1)
plt.xlabel('Temperature [K]', fontsize=14)
plt.ylabel('Hyd. Ionization Fraction', fontsize=14)
plt.title(f'Hyd. Number Density: {numden:.2e} m$^{{-3}}$')
