#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 13:50:06 2024

@author: konstantinos
"""

import numpy as np
import healpy as hp

def find_sph_coord(theta,phi,r=1):
    x = r*np.sin(np.pi-theta) * np.cos(phi) #because theta starts from the z axis: we're flipped
    y = r*np.sin(np.pi-theta) * np.sin(phi)
    z = r*np.cos(np.pi-theta)
    #xyz = [x, y, z]
    return x,y,z


def select_observer(wanted_theta, wanted_phi, thetas, phis):
    x_hp = np.zeros(192)
    y_hp = np.zeros(192)
    z_hp = np.zeros(192)
    wanted_xyz = find_sph_coord(wanted_theta, wanted_phi)
    
    # Healpix XYZ
    for i in range(192): 
        x_hp[i], y_hp[i], z_hp[i] = find_sph_coord(thetas[i], phis[i])
        
    observers = np.array([x_hp, y_hp, z_hp]).T
    inner_product = [np.dot(wanted_xyz, observer) for observer in observers]
    index = np.argmax(inner_product) 
        
    return index

# Find observers with Healpix 
thetas = np.zeros(192)
phis = np.zeros(192) 
observers = []
xyz_grid = []
stops = np.zeros(192) 
for iobs in range(0,192):
    theta, phi = hp.pix2ang(4, iobs) # theta in [0,pi], phi in [0,2pi]
    thetas[iobs] = theta
    phis[iobs] = phi
        
wanted_thetas = [np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi, 0] # x, -x, y, -y, z, -z
wanted_phis = [0, np.pi, np.pi/2, 3*np.pi/2, 0, 0]
wanted_indices = []
for idx in range(len(wanted_thetas)):
    wanted_theta = wanted_thetas[idx]
    wanted_phi = wanted_phis[idx]
    wanted_index = select_observer(wanted_theta, wanted_phi, thetas, phis)
    wanted_indices.append(wanted_index)