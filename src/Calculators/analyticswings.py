#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:19:55 2024

@author: konstantinos
"""
import numpy as np
import sympy as sym 
import matplotlib.pyplot as plt

# Symbols
theta, e, M, mstar, rstar = sym.symbols(r'\theta e M m_* r_*')
Rt = rstar * (M/mstar)**(1/3)
tfb = np.pi / np.sqrt(2) * (rstar**3/mstar)**(1/2) * (M/mstar)**(1/2)
jp = sym.sqrt(2*Rt*M)
r = jp**2 / M * 1/(1+sym.cos(theta))
a_g = M/r**2
a_T = 2*M/r**3
a_sg = 2*mstar/rstar
L = r**(-1/2)
a_p = mstar/rstar**2 * (rstar/L)**(2-5/3)

classic_de = 2 * mstar/rstar * ((M/mstar)**(1/3) + 1)
print(f'classic DE: {classic_de.subs({M:1e6, mstar:1, rstar:1}):.2f}')
#%%
e = 1 - (mstar/M)**(1/3)
r = jp**2 / M * 1/(1+e*sym.cos(theta))
dr = sym.diff(r, theta)
rho0 = mstar/(2*np.pi*rstar**3)
rho = rho0 * Rt**(0.5) * r**(1/2)
Fsg = rho0 * Rt**2 * 2*np.pi / rstar * r**(-2)
Ft = 4*M*rstar*Rt**(-0.5) * r**(-2.5)
Fp = mstar*Rt*2**(-7/3)*r**(-2)*r

#%%
start = 0
end = np.pi
ends = np.linspace(0.1, np.pi, 10)
SGs = np.zeros((3,10))
Ps = np.zeros((3,10))
Mbhs = [1e4, 1e5, 1e6]
for j, end in enumerate(ends):
    for i, Mbh in enumerate(Mbhs):
        classic_de = 2 * 0.5/0.47 * ((Mbh/0.5)**(1/3) + 1)
        integral_SG = sym.integrate(Fsg*dr, (theta, start, end) )
        integral_P = sym.integrate(Fp*dr, (theta, start, end) )
        Wsg = integral_SG.subs({M:Mbh, mstar:0.5, rstar:0.5})
        Wp = integral_P.subs({M:Mbh, mstar:0.5, rstar:0.5})
        SGs[i][j] = Wsg / classic_de
        Ps[i][j] = Wp / classic_de
#%%
plt.figure(figsize = (3,3), dpi = 300)
ls = ['-', '--', ':']
for i, Mbh in enumerate(Mbhs):
    plt.plot(ends, SGs[i], c='k', ls = ls[i], marker = 'o', markersize = 2)
    plt.plot(ends, Ps[i], c='royalblue', ls = ls[i], marker = 'o', markersize = 2)
    if i == 0:
        plt.plot(ends, SGs[i], c='k', label = 'Self Grav.' , ls = ls[i])
        plt.plot(ends, Ps[i], c='royalblue', label = 'Pressure', ls = ls[i])

plt.xlabel('True Anomaly')
plt.ylabel('Work/$\Delta E$')
#plt.xlim(0, np.pi/2)
plt.legend()


