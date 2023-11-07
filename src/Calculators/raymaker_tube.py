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
    fig, ax = plt.subplots(1,1, subplot_kw=dict(projection="mollweide"))
    for i in range(192):
        # Get Observers
        thetas[i], phis[i] = hp.pix2ang(NSIDE, i)
        thetas[i] -= np.pi/2 # Enforce theta in -pi to pi    
        observers.append( (thetas[i], phis[i]) )
        
        # Get their opening angles
        theta_diff = 0
        phi_diff = 0
        corner_angles = []
        
        # Corner XYZ coordinates
        corner_x, corner_y, corner_z = hp.boundaries(4,i)
        corners = np.zeros((4,3))
        corner_angles = np.zeros((4,2))
        
        # Corner 
        for i in range(4):
            corners[i] = corner_x[i], corner_y[i], corner_z[i]
            corner_angles[i][0] = hp.vec2ang(corners[i])[0][0]
            corner_angles[i][1] = hp.vec2ang(corners[i])[1][0]
            
        # Biggest possible Delta phi and Delta theta
        # print(corner_angles.T[1].max() - corner_angles.T[1].min())
        corner_angles_phi_plot = list(corner_angles.T[1] - np.pi)
        corner_angles_theta_plot = list(corner_angles.T[0] - np.pi/2)
        
        # last point to close the rectangle
        corner_angles_phi_plot.append(corner_angles_phi_plot[0]) 
        corner_angles_theta_plot.append(corner_angles_theta_plot[0])
        
        if plot:
            plt.plot(corner_angles_phi_plot, corner_angles_theta_plot, '-x', 
                      markersize = 1, c='k', linewidth = 1, zorder = 4,)
        for i in range(0,4):
            theta_diff_temp = np.max(np.abs(corner_angles[i][0] - corner_angles[:][0]))
            phi_diff_temp = np.max(np.abs(corner_angles[i][1] - corner_angles[:][1])) 
            
            if theta_diff_temp > theta_diff:
                theta_diff = theta_diff_temp
                
            if phi_diff_temp > phi_diff:
                phi_diff = phi_diff_temp
                
        delta_thetas.append(theta_diff/2)
        delta_phis.append(phi_diff/2)

    # Simulation Points plot
    if plot:
        plt.plot(PHI[::100] - np.pi, THETA[::100], 
                 'x',  c = 'r', markersize = 0.1, zorder = 2) 
    return THETA, PHI, R, T, Den, Rad, observers

@numba.njit
def ray_maker_doer(THETA, PHI, R, T, Den, Rad, observers):
    # Make rays
    rays_T = []
    rays_Den = []
    rays_R = []
    rays_Rad = []
    
    # NOTE: These are kind of wrong, we should be more explicit about this
    delta_phi = 0.392 / 2 # rad
    delta_theta = 0.339 / 2 # rad
    for observer in observers:

        # numpy thinks in C
        theta_mask = ( observer[0] - delta_theta < THETA) & \
                      ( THETA < observer[0] + delta_theta)
                     
        phi_mask = ( observer[1] - delta_phi < PHI) & \
                    ( PHI < observer[1] + delta_phi)
        fluff_mask = ( Den > 1e-17 )
        
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
    
    return rays_T, rays_Den, rays_Rad, rays_R
        
#         # These are kind of wrong, we should be more explicit about this
#         delta_phi = 0.392 / 2 # rad
#         delta_theta = 0.339 / 2 # rad
#         # numpy thinks in C
#         theta_mask = ( observer[0] - delta_theta < THETA) & \
#                      ( THETA < observer[0] + delta_theta)
                     
#         phi_mask = ( observer[1] - delta_phi < PHI) & \
#                    ( PHI < observer[1] + delta_phi)
#         fluff_mask = ( Den > 1e-17 )
#         #
#         # Mask & Mass Weigh
#         # ray_Mass = Mass[theta_mask & phi_mask]
#         # sum_mass = np.sum(ray_Mass)
#         ray_R = R[theta_mask & phi_mask & fluff_mask] 
#         ray_Den = Den[theta_mask & phi_mask  & fluff_mask ]# * ray_Mass / sum_mass
#         ray_T = T[theta_mask & phi_mask  & fluff_mask]# * ray_Mass / sum_mass
#         ray_Rad = Rad[theta_mask & phi_mask  & fluff_mask]# * ray_Mass / sum_mass
#         #
#         # Sort by r
#         bookeeper = np.argsort(ray_R)
#         ray_R = ray_R[bookeeper]
#         ray_Den = ray_Den[bookeeper]
#         ray_T = ray_T[bookeeper]
#         ray_Rad = ray_Rad[bookeeper]
#         #
#         # Keep
#         rays_R.append(ray_R)
#         rays_Den.append( ray_Den )
#         rays_T.append( ray_T)
#         rays_Rad.append(ray_Rad)
def ray_maker(fix, m, prune = 1, plot = False):
    THETA, PHI, R, T, Den, Rad, observers = loader(fix, m, prune, plot)
    rays_T, rays_Den, rays_Rad, rays_R = ray_maker_doer(THETA, PHI, R, T, Den, 
                                                        Rad, observers)
    
    return rays_T, rays_Den, rays_Rad, rays_R

def isalice():
    return alice
if __name__ == '__main__':
    # ray_maker('844', 6, prune = 1 , plot = True)
    isalice()