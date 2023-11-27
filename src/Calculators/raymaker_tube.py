#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 17:55:00 2023

@author: konstantinos
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 16:22:40 2023

@author: konstantinos
"""
alice = False
    
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from astropy.coordinates import cartesian_to_spherical
import numba

# Constants
NSIDE = 4
G = 6.6743e-11 # SI
Msol = 1.98847e30 # kg
Rsol = 6.957e8 # m
t = np.sqrt(Rsol**3 / (Msol*G )) # Follows from G = 1
c = 3e8 * t/Rsol # simulator units. Need these for the PW potential
c_cgs = 3e10 # [cm/s]
Msol_to_g = 1.989e33 # [g]
Rsol_to_cm = 6.957e10 # [cm]
den_converter = Msol_to_g / Rsol_to_cm**3
en_den_converter = Msol_to_g / (Rsol_to_cm  * t**2 ) # Energy Density converter

def loader(fix, m, prune = 1, plot = False):
    fix = str(fix)
    if plot:
           fig = plt.figure()
           ax = fig.add_subplot(111,) #projection='mollweide')
    if alice:
        pre = '/home/s3745597/data1/TDE/'
        # Import
        X = np.load(pre + str(m) + '/snap_'  + fix + '/CMx_' + fix + '.npy')
        Y = np.load(pre + str(m) + '/snap_'  + fix + '/CMy_' + fix + '.npy')
        Z = np.load(pre + str(m) + '/snap_'  + fix + '/CMz_' + fix + '.npy')
        Mass = np.load(pre + str(m) + '/snap_'  + fix + '/Mass__' + fix + '.npy')
        T = np.load(pre + str(m) + '/snap_'  + fix + '/T_' + fix + '.npy')
        Den = np.load(pre + str(m) + '/snap_'  + fix + '/Den_' + fix + '.npy')
        Rad = np.load(pre + str(m) + '/snap_'  +fix + '/Rad_' + fix + '.npy') 
    else:
        X = np.load( str(m) + '/'  + fix + '/CMx_' + fix + '.npy')[::prune]
        Y = np.load( str(m) + '/'  + fix + '/CMy_' + fix + '.npy')[::prune]
        Z = np.load( str(m) + '/'  + fix + '/CMz_' + fix + '.npy')[::prune]
        Mass = np.load( str(m) + '/'  + fix + '/Mass_' + fix + '.npy')[::prune]
        T = np.load( str(m) + '/'  + fix + '/T_' + fix + '.npy')[::prune]
        Den = np.load( str(m) + '/'  + fix + '/Den_' + fix + '.npy')[::prune]
        Rad = np.load( str(m) + '/'  +fix + '/Rad_' + fix + '.npy')[::prune]
    
    # Convert to CGS
    Rad *= Den 
    Rad *= en_den_converter
    Den *= den_converter 

    # Convert to spherical
    R, THETA, PHI = cartesian_to_spherical(X,Y,Z)
    R = R.value 
    THETA = THETA.value
    PHI = PHI.value

    # Find observers with Healpix
    thetas = np.zeros(192)
    phis = np.zeros(192)
    observers = []
    delta_thetas = []
    delta_phis = []
    for i in range(192):
        # Get Observers
        thetas[i], phis[i] = hp.pix2ang(4, i)
        thetas[i] -= np.pi/2 # Enforce theta in -pi to pi    
        observers.append( (thetas[i], phis[i]) )
        
        # Get their opening angles
        corner_angles = []
        # Corner XYZ coordinates
        corner_x, corner_y, corner_z = hp.boundaries(4,i)
        corners = np.zeros((4,3))
        corner_angles = np.zeros((4,2))
        
        # Corner 
        for l in range(4):
            corners[l] = corner_x[l], corner_y[l], corner_z[l]
            corner_angles[l][0] = hp.vec2ang(corners[l])[0][0]
            corner_angles[l][1] = hp.vec2ang(corners[l])[1][0]

        for j in range(4):
            if corner_angles[j][1] > 3/2 * np.pi:
                for k in range(4):
                    if corner_angles[k][1] < np.pi:
                        corner_angles[k][1] = 2 * np.pi - 0.001

        corner_angles_phi_plot = list(corner_angles.T[1] - np.pi)
        corner_angles_theta_plot = list(corner_angles.T[0] - np.pi/2)
        
        
        # last point to close the rectangle
        corner_angles_phi_plot.append(corner_angles_phi_plot[0]) 
        corner_angles_theta_plot.append(corner_angles_theta_plot[0])

        if plot:
            ax.plot(corner_angles_phi_plot, corner_angles_theta_plot, '-h', 
                      markersize = 5, c='k', linewidth = 0.3, zorder = 4,)
        
        
    # Simulation Points plot
    if plot:
        ax.plot(PHI[::20] - np.pi, THETA[::20], 
                  'x',  c = AEK, markersize = 0.1, zorder = 2) 
        # ax.set_xticks([])
        # ax.set_yticks([])
    return THETA, PHI, R, T, Den, Rad, observers, delta_thetas, delta_phis

@numba.njit
def ray_maker_doer(THETA, PHI, R, T, Den, Rad, observers, 
                   delta_thetas, delta_phis):
    # Make rays
    rays_T = []
    rays_Den = []
    rays_R = []
    rays_Rad = []
    
    for i, observer in enumerate(observers):
        delta_theta = delta_thetas[i] # 0.332 / 2 
        delta_phi = delta_phis[i] # 0.339 / 2 
        
        
        # numpy thinks in C
        theta_mask = ( observer[0] - delta_theta < THETA) & \
                      ( THETA < observer[0] + delta_theta)
                     
        phi_mask = ( observer[1] - delta_phi < PHI) & \
                    ( PHI < observer[1] + delta_phi)
        fluff_mask = ( Den > 1e-17 ) # 1e-17 works
 
        # Mask
        ray_R = R[theta_mask & phi_mask & fluff_mask] 
        ray_Den = Den[theta_mask & phi_mask  & fluff_mask ]# * ray_Mass / sum_mass
        ray_T = T[theta_mask & phi_mask  & fluff_mask]# * ray_Mass / sum_mass
        ray_Rad = Rad[theta_mask & phi_mask  & fluff_mask]# * ray_Mass / sum_mass
        
        # Sort by r
        bookeeper = np.argsort(ray_R)
        ray_R = ray_R[bookeeper]
        ray_Den = ray_Den[bookeeper]
        ray_T = ray_T[bookeeper]
        ray_Rad = ray_Rad[bookeeper]
    
        # Keep
        rays_R.append(ray_R)
        rays_Den.append( ray_Den )
        rays_T.append( ray_T)
        rays_Rad.append(ray_Rad)
        break
    return rays_T, rays_Den, rays_Rad, rays_R
        

def ray_maker(fix, m, prune = 1, plot = False):
    THETA, PHI, R, T, Den, Rad, observers, delta_thetas, delta_phis = loader(fix, m, prune, plot)
    rays_T, rays_Den, rays_Rad, rays_R = ray_maker_doer(THETA, PHI, R, T, Den, 
                                                        Rad, observers, delta_thetas, delta_phis)
    
    return rays_T, rays_Den, rays_Rad, rays_R

def isalice():
    return alice
if __name__ == '__main__':
    plt.rcParams['text.usetex'] = True
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['figure.figsize'] = [5 , 4]
    plt.rcParams['axes.facecolor']= 	'whitesmoke'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    AEK = '#F1C410' # Important color 
    loader('844', 6, prune = 1 , plot = True)
    isalice()