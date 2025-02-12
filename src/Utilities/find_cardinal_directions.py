#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 13:50:06 2024

@author: konstantinos
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import src.Utilities.prelude as c

def find_sph_coord(theta,phi,r=1):
    x = r*np.sin(np.pi-theta) * np.cos(phi) #because theta starts from the z axis: we're flipped
    y = r*np.sin(np.pi-theta) * np.sin(phi)
    z = r*np.cos(np.pi-theta)
    #xyz = [x, y, z]
    return x,y,z


def select_observer(wanted_theta, wanted_phi, thetas, phis):
    x_hp = np.zeros(c.NPIX)
    y_hp = np.zeros(c.NPIX)
    z_hp = np.zeros(c.NPIX)
    wanted_xyz = find_sph_coord(wanted_theta, wanted_phi)
    
    # Healpix XYZ
    for i in range(c.NPIX): 
        x_hp[i], y_hp[i], z_hp[i] = find_sph_coord(thetas[i], phis[i])
        
    observers = np.array([x_hp, y_hp, z_hp]).T
    inner_product = [np.dot(wanted_xyz, observer) for observer in observers]
    index = np.argmax(inner_product) 
        
    return index

# Find observers with Healpix 
thetas = np.zeros(c.NPIX)
phis = np.zeros(c.NPIX) 
observers = []
xyz_grid = []
stops = np.zeros(c.NPIX) 
for iobs in range(0,c.NPIX):
    theta, phi = hp.pix2ang(c.NSIDE, iobs) # theta in [0,pi], phi in [0,2pi]
    thetas[iobs] = theta
    phis[iobs] = phi
        
# Cardinal
wanted_thetas = [np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi, 0] # x, -x, y, -y, z, -z
wanted_phis = [0, np.pi, np.pi/2, 3*np.pi/2, 0, 0]
wanted_indices = []

# Z-sweep
# wanted_thetas = np.linspace(0.5, 1, num = 6) * np.pi
# wanted_phis = np.zeros(len(wanted_thetas))
# wanted_indices = []

for idx in range(len(wanted_thetas)):
    wanted_theta = wanted_thetas[idx]
    wanted_phi = wanted_phis[idx]
    wanted_index = select_observer(wanted_theta, wanted_phi, thetas, phis)
    wanted_indices.append(wanted_index)
    
    
# Check, to make sure
fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132)
ax2.set_title('X-Z')
ax3 = fig.add_subplot(133)
ax3.set_title('X-Y')


ax1.scatter(0,0,0, marker = 's', color = 'k')
ax2.scatter(0,0, marker = 's', color = 'k')
ax3.scatter(0,0, marker = 's', color = 'k')

colors = [c.darkb, c.cyan, c.prasinaki, c.yellow, c.kroki, c.reddish]
wanted_indices = np.arange(0, c.NPIX)
zsweep = [104, 136, 152, 167, 168, 179, 180, 187, 188, 191]#, 140]

# zsweep = [72, 104, 136, 168, 180, 188]
# idea = [188, 180, 168, 152, 
#         136, 104,
#         191, 187, 179, 167, 
#         140]
angles = []
for i in zip(zsweep):#, colors):
    x, y, z = find_sph_coord(thetas[i], phis[i])
    # if i[0] in zsweep:
    if np.sqrt(x**2 + z**2) > 0.5 and x > 0 and z > 0: # and i[0] in idea:
        i = int(i[0])
        angle = np.arctan2(z,x) * 180 / np.pi
        print(i, angle)
        angles.append(angle)
        ax1.scatter(x,y,z, marker = f'${i}$', s = 200)#, color = col)
        ax2.scatter(x, z, marker = f'${i}$', s = 500, c = 'k')#, color = col)
        ax3.scatter(x, y, marker = f'${i}$', s = 500, c = 'k')#, color = col)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax2.set_xlabel('x')
ax2.set_ylabel('z')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
plt.tight_layout()



