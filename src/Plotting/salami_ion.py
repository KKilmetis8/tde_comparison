#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 16:29:01 2024

@author: konstantinos
"""

import numpy as np
import matplotlib.pyplot as plt
import colorcet

import src.Utilities.prelude

rstar = 0.47
mstar = 0.5
Mbh = 10000
extra = 'beta1S60n1.5Compton'
extra2 = 'beta1S60n1.5ComptonHiRes'
simname = f'R{rstar}M{mstar}BH{Mbh}{extra}' 
simname2 = f'R{rstar}M{mstar}BH{Mbh}{extra2}' 

pre = 'data/'
salami_trans = np.loadtxt(f'{pre}ion{simname}.txt')
days = np.loadtxt(f'{pre}ef8/eccdays{simname}.txt')
# ecc2 = np.loadtxt(f'{pre}ecc{simname2}.txt')
# days2 = np.loadtxt(f'{pre}eccdays{simname2}.txt')

Rt = rstar * (Mbh/mstar)**(1/3) # Msol = 1, Rsol = 1
apocenter = Rt * (Mbh/mstar)**(1/3)

ray_no = 200
thetas = np.linspace(-np.pi, np.pi, num = ray_no)
theta_trans = np.zeros(len(salami_trans))
for i in range(len(salami_trans)):
    theta_trans[i] = int(salami_trans[i][0])
#%%
fig, ax = plt.subplots(1,1, figsize = (4,4))

img1 = ax.plot(theta_trans, c = 'k', lw = 2, marker = 'h')

ax.set_xlabel('Time [t/$t_\mathrm{FB}$]')
ax.set_ylabel('Transition Salami')
ax.set_title(r'$10^4$ M$_\odot$')