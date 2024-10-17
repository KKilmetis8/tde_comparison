#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 11:28:57 2024

@author: konstantinos
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 12:38:03 2024

@author: konstantinos
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import colorcet
import src.Utilities.prelude as c
m = 6
Mbh = 10**m
pre = f'{m}/'
snap = 351
rstar = 0.47
mstar = 0.5
Rt = rstar * (Mbh/mstar)**(1/3)
X = np.load(f'{pre}{snap}/CMx_{snap}.npy')
Y = np.load(f'{pre}{snap}/CMy_{snap}.npy')
Z = np.load(f'{pre}{snap}/CMz_{snap}.npy')
VX = np.load(f'{pre}{snap}/Vx_{snap}.npy')
VY = np.load(f'{pre}{snap}/Vy_{snap}.npy')
VZ = np.load(f'{pre}{snap}/Vz_{snap}.npy')
Den = np.load(f'{pre}{snap}/Den_{snap}.npy')
day = np.loadtxt(f'{pre}{snap}/tbytfb_{snap}.txt')
#%%
denmask = Den > 1e-13
X = X[denmask]
Y = Y[denmask]
Z = Z[denmask]
VX = VX[denmask]
VY = VY[denmask]
VZ = VZ[denmask]
Den = Den[denmask]

R = np.sqrt(X**2 + Y**2 + Z**2)
V = np.sqrt(VX**2 + VY**2 + VZ**2)
rg = 2*Mbh/c.c**2
Orb = 0.5*V**2 - Mbh / (R-rg) 
bound_mask = Orb < 0
Orb = Orb[bound_mask]
X = X[bound_mask]
Y = Y[bound_mask]
Z = Z[bound_mask]
VX = VX[bound_mask]
VY = VY[bound_mask]
VZ = VZ[bound_mask]
Den = Den[bound_mask]

denplot = Den * c.den_converter 
denplot = denplot / np.max(denplot)

Vvec = np.array([VX, VY, VZ]).T
Rvec = np.array([X, Y, Z]).T
jvec = np.cross(Rvec, Vvec)
j = [np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2) for vec in jvec]
ecc = [np.sqrt(1 + 2*jt**2*energy/Mbh**2) for jt, energy in zip(j, Orb)]
#%%
Rt = rstar * (Mbh/mstar)**(1/3)
Rp = Rt
jp = np.sqrt(2*Rp*Mbh)

delta_energy_max = mstar/rstar * ( (Mbh/mstar)**(1/3) - 1 )
ecirc = Mbh/(4*Rt)
#%%
fig = plt.figure(figsize=(4,4))

def theory(mbh,j):
    return mbh**2/(2*j**2)
theory_js = np.logspace(-5, 6, 1000)
theory_es = [ theory(Mbh, j) for j in theory_js]
theory_es = np.array(theory_es)/delta_energy_max
plt.plot(theory_es, theory_js/jp, c='k', lw = 4, zorder = 2, alpha = 0.2)
plt.xscale('log')
plt.yscale('log')

step = 100
plt.scatter(-Orb[::step]/delta_energy_max, j[::step]/jp, c=ecc[::step], 
            alpha=denplot[::step],
            s = 1, cmap = 'cet_rainbow4', vmin = 0, vmax = 1)
plt.axhline(jp/jp, color = 'k', ls = '--')
plt.axvline(ecirc/delta_energy_max, color = 'k', ls='--')
cb = plt.colorbar()

plt.xlim(1e-3, ecirc/delta_energy_max*100 )
plt.ylim(1e-1, 1e1)
plt.xlabel('Orbital Energy $[\Delta E_\mathrm{max}]$')
plt.ylabel('Angular Momentum $[j_\mathrm{par}]$')


plt.text(0.16, 0.17, f'{day:.2f} $t_\mathrm{{FB}}$', fontsize = 13, 
         transform = fig.transFigure,
         bbox=dict(facecolor='whitesmoke', edgecolor='black'))
plt.text(0.48, 0.16, r'$E_\mathrm{circ}$', fontsize = 10, 
         transform = fig.transFigure)
plt.text(0.2, 0.80, r'$j = M_\mathrm{BH} \epsilon^{-1/2}$', fontsize = 10, 
         transform = fig.transFigure,
         bbox=dict(facecolor='whitesmoke', edgecolor='black'))
cb.set_label('Eccentricity')
plt.title(f'Action Space $|$ $10^{m} M_\odot$')


