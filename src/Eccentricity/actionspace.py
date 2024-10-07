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
from src.Utilities.parser import parse
pre = '/home/s3745597/data1/TDE/'
args = parse()
sim = args.name
Mbh = args.blackhole
snap = args.only
print(snap)
Mbh = float(Mbh)
rstar = 0.47
mstar = 0.5
X = np.load(f'{pre}{sim}/snap_{snap}/CMx_{snap}.npy')
Y = np.load(f'{pre}{sim}/snap_{snap}/CMy_{snap}.npy')
Z = np.load(f'{pre}{sim}/snap_{snap}/CMz_{snap}.npy')
VX = np.load(f'{pre}{sim}/snap_{snap}/Vx_{snap}.npy')
VY = np.load(f'{pre}{sim}/snap_{snap}/Vy_{snap}.npy')
VZ = np.load(f'{pre}{sim}/snap_{snap}/Vz_{snap}.npy')
Den = np.load(f'{pre}{sim}/snap_{snap}/Den_{snap}.npy')
# day = np.loadtxt(f'{pre}{snap}/tbytfb_{snap}.txt')
#%%
denmask = Den > 1e-18
X = X[denmask]
Y = Y[denmask]
Z = Z[denmask]
VX = VX[denmask]
VY = VY[denmask]
VZ = VZ[denmask]
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
Vvec = np.array([VX, VY, VZ]).T
Rvec = np.array([X, Y, Z]).T
jvec = np.cross(Rvec, Vvec)
j = [np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2) for vec in jvec]
ecc = [np.sqrt(1 + 2*jt**2*energy/Mbh**2) for jt, energy in zip(j, Orb)]
#%%
savepath = f'{pre}/tde_comparison/data/actionspace/{sim}/'
np.save(f'{savepath}j_{snap}', j)
np.save(f'{savepath}orb_{snap}', Orb)
np.save(f'{savepath}ecc_{snap}', ecc)




