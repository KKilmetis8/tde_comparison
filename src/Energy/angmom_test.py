#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:54:50 2025

@author: konstantinos

local tests for angular momentum conservation and energy split
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.Utilities.loaders import local_loader
import src.Utilities.prelude as c

fig, ax = plt.subplots()
ms = [4, 5, 6]
cols = ['k', c.AEK, 'maroon']
# Ang mom
for m, col in zip(ms, cols):
    Mbh = 10**m
    rg = 2*Mbh / c.c**2
    if m == 4:
        localfixes = [116, 136, 164, 179, 199, 218, 240, 272, 297, 300, 348]
    if m == 5:
        localfixes = [227, 236, 288, 301, 308, 349]
    if m == 6:
        localfixes = [180, 290, 315, 325, 351, 379, 444]
    jsums = []
    ts = []
    
    for fix in tqdm(localfixes):
        X, Y, Z, Vx, Vy, Vz, Mass, t = local_loader(m, fix, 'orbital+mass')
        ts.append(t)
        R = np.sqrt(np.power(X, 2) + np.power(Y, 2) + np.power(Z, 2))
        V = np.sqrt(np.power(Vx, 2) + np.power(Vy, 2) + np.power(Vz, 2))
        Orbital = (0.5 * V**2) - Mbh / (R-rg)
        bound_mask = Orbital < 0
        rvec = np.array([X[bound_mask], Y[bound_mask], Z[bound_mask]])
        vvec = np.array([Vx[bound_mask], Vy[bound_mask], Vz[bound_mask]]) * Mass[bound_mask] 
        jvec = np.cross(rvec.T, vvec.T)
        jmag = np.linalg.norm(jvec)
        jsums.append(np.sum(jmag))

    ax.plot(ts, jsums, '-o', c = col, lw = 0.75, markersize = 2,
            label = f'10$^{m}$ M$_\odot$') 
    
ax.set_xlim(1,)
ax.legend(frameon = False)
ax.set_yscale('log')
ax.set_xlabel('Time [$t_\mathrm{FB}$]', fontsize = 10)
ax.set_ylabel(r'Sum of Angular Momentum [$M_\odot R_\odot^2/\tilde{t}]$',
              fontsize = 10)

#%%
from matplotlib.patches import Polygon
fig, ax = plt.subplots()

m = 4
localfixes = [300]
Rt = 0.47 * (10**m/0.5)**(1/3)
for fix in tqdm(localfixes):
    X, Y, Z, Vx, Vy, Vz, Mass, t = local_loader(m, fix, 'orbital+mass')
    plt.scatter(X[::1000]/Rt, Y[::1000]/Rt, c = 'k')
pts = np.array([ [-2*Rt, 2*Rt,], [2*Rt, 2*Rt], [2*Rt, -2*Rt], [-2*Rt, 2*Rt]])
p = Polygon(pts)
ax.add_patch(p)
    
