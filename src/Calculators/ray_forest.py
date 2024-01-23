#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gives ray which are logspaced. Around photosphere they sould be 1.
Created on Tue Oct 10 10:19:34 2023

@author: paola, konstantinos

"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

import numpy as np
from scipy.spatial import KDTree
import healpy as hp
from src.Luminosity.select_path import select_prefix
import matplotlib.pyplot as plt
AEK = '#F1C410'
from src.Utilities.isalice import isalice
alice, plot = isalice()
#%% Constants & Converter
NSIDE = 4
G = 6.6743e-11 # SI
Msol = 2e30 #1.98847e30 # kg
Rsol = 7e8 #6.957e8 # m
t = np.sqrt(Rsol**3 / (Msol*G )) # Follows from G = 1
#c = 3e8 * t/Rsol # simulator units. Need these for the PW potential
c_cgs = 3e10 # [cm/s]
Msol_to_g = 2e33 #1.989e33 # [g]
Rsol_to_cm = 7e10 #6.957e10 # [cm]
den_converter = Msol_to_g / Rsol_to_cm**3
en_den_converter = Msol_to_g / (Rsol_to_cm  * t**2 ) # Energy Density converter

def isalice():
    return alice

def find_sph_coord(r, theta,phi):
    x = r * np.sin(theta) * np.cos(phi) #because theta should start from the z axis: we're flipped
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return [x,y,z]

def ray_maker(fix, m, check, thetas, phis, stops, num): 
    """ 
    Num is 1001 because for blue we then delete the last cell.
    Outputs are in CGS with exception of ray_vol (in solar units).
    """
    fix = str(fix)
    Mbh = 10**m 
    Rt =  Mbh**(1/3) # Msol = 1, Rsol = 1
    apocenter = 2 * Rt * Mbh**(1/3)

    # Load data
    pre = select_prefix(m, check)
    X = np.load(pre + fix + '/CMx_' + fix + '.npy')
    Y = np.load(pre + fix + '/CMy_' + fix + '.npy')
    Z = np.load(pre + fix + '/CMz_' + fix + '.npy')
    VX = np.load(pre + fix + '/Vx_' + fix + '.npy')
    VY = np.load(pre + fix + '/Vy_' + fix + '.npy')
    VZ = np.load(pre + fix + '/Vz_' + fix + '.npy')
    T = np.load(pre + fix + '/T_' + fix + '.npy')
    Den = np.load(pre + fix + '/Den_' + fix + '.npy')
    Rad = np.load(pre + fix + '/Rad_' + fix + '.npy')
    IE = np.load(pre + fix + '/IE_' + fix + '.npy')
    Vol = np.load(pre + fix + '/Vol_' + fix + '.npy')
    Tcool_min = np.loadtxt('src/Opacity/Tcool_ext.txt')[0]
    
    # Move pericenter to 0
    # X -= Rt
    # Convert Energy / Mass to Energy Density in CGS
    Rad *= Den 
    Rad *= en_den_converter
    IE *= Den 
    IE *= en_den_converter
    Den *= den_converter 
    
    # make a tree
    sim_value = [X, Y, Z] 
    sim_value = np.transpose(sim_value) #array of dim (number_points, 3)
    sim_tree = KDTree(sim_value) 
    
    # # Find observers with Healpix
    # thetas = np.zeros(192)
    # phis = np.zeros(192) 
    # observers = []
    # for i in range(0,192):
    #     thetas[i], phis[i] = hp.pix2ang(NSIDE, i) # theta in [0,pi], phi in [0,2pi]
    #     #phis[i] -= np.pi # Enforce theta in -pi to pi for astropy
    #     observers.append( (thetas[i], phis[i]) )

    # Ensure that the regular grid cells are smaller than simulation cells
    start = 0.56 #1e-0.25 Solar radii # 0.1* Rt 
    rays_radii = []

    tree_indexes = np.zeros((len(thetas), num-1))
    # you take num-1 beacause in blue you will delete the last cell of radii
    rays_T = np.zeros((len(thetas), num-1))
    rays_den = np.zeros((len(thetas), num-1))
    rays = np.zeros((len(thetas), num-1))
    rays_ie = np.zeros((len(thetas), num-1))
    rays_vol = np.zeros((len(thetas), num-1))
    rays_v = np.zeros((len(thetas), num-1))

    for j in range(len(thetas)):
        stop =  stops[j]
        log_start = np.log10(start)
        log_stop = np.log10(stop)
        log_radii = np.linspace(log_start, log_stop, num) #simulator units
        radii = 10**log_radii
        
        for k in range(len(radii)-1):
            radius = radii[k]
            queried_value = find_sph_coord(radius, thetas[j], phis[j])
            queried_value[0] += Rt
            _, idx = sim_tree.query(queried_value)

            # Store
            tree_indexes[j][k] = idx 
            rays_T[j][k] = max(T[idx], Tcool_min)
            rays_den[j][k] = Den[idx] 
            rays[j][k] = Rad[idx] 
            rays_ie[j][k] = IE[idx] 
            rays_vol[j][k] = Vol[idx] # not in CGS
            vel = VX[idx]**2 + VY[idx]**2 + VZ[idx]**2 #np.sqrt(VX[idx]**2 + VY[idx]**2 + VZ[idx]**2)
            vel *= Rsol_to_cm / t #convert in CGS
            rays_v[j][k] = vel

        # Convert to CGS
        radii *= Rsol_to_cm
        rays_radii.append(radii)
    
    # Remove Bullshit
    rays = np.nan_to_num(rays, neginf = 0)
    rays_ie = np.nan_to_num(rays, neginf = 0)
    rays_den = np.nan_to_num(rays_den, neginf = 0)
    rays_T = np.nan_to_num(rays_T, neginf = 0)

    #%% Plot
    # ax = plt.figure().add_subplot(projection='3d')
    # radii = np.array(radii)
    # ax.scatter(our_x[::10], our_y[::10], our_z[::10], 
    #             c = radii[::10], cmap = 'cet_bmy', alpha = 0.18, zorder = 2)
    # # Selecting one ray
    # pick = 107
    # rat = pick * num
    # old_rat = rat - num
    # r_rat = np.sqrt(our_x[old_rat:rat]**2 + our_y[old_rat:rat]**2 + our_z[old_rat:rat]**2)
    # ax.scatter(our_x[old_rat:rat], our_y[old_rat:rat], our_z[old_rat:rat], 
    #             c = 'k', alpha = 1, zorder = 10)
    # ax.set_xlim(-10_000, 10_000)
    # ax.set_ylim(-10_000, 10_000)
    # ax.set_zlim(-10_000, 10_000)

    return tree_indexes, rays_T, rays_den, rays, rays_ie, rays_radii, rays_vol, rays_v

 
if __name__ == '__main__':
    import h5py
    m = 6
    num = 1000
    snap = 882
    check = 'fid'
    filename = f"{m}/{snap}/snap_{snap}.h5"

    box = np.zeros(6)
    with h5py.File(filename, 'r') as fileh:
        for i in range(len(box)):
            box[i] = fileh['Box'][i]
    # print('Box', box)

    # Find observers with Healpix 
    # For each of them set the upper limit for R
    thetas = np.zeros(192)
    phis = np.zeros(192) 
    observers = []
    stops = np.zeros(192) 
    for iobs in range(0,192):
        theta, phi = hp.pix2ang(4, iobs) # theta in [0,pi], phi in [0,2pi]
        thetas[iobs] = theta
        phis[iobs] = phi
        observers.append( (theta, phi) )
        xyz = find_sph_coord(1, theta, phi)
        mu_x = xyz[0]
        mu_y = xyz[1]
        mu_z = xyz[2]

        # Box is for 
        if(mu_x < 0):
            rmax = box[0] / mu_x
        else:
            rmax = box[3] / mu_x
        if(mu_y < 0):
            rmax = min(rmax, box[1] / mu_y)
        else:
            rmax = min(rmax, box[4] / mu_y)
        if(mu_z < 0):
            rmax = min(rmax, box[2] / mu_z)
        else:
            rmax = min(rmax, box[5] / mu_z)

        stops[iobs] = rmax
    print(np.max(stops))

    tree_indexes, rays_T, rays_den, rays, rays_ie, rays_radii, rays_vol, rays_v = ray_maker(snap, m, check, thetas, phis, stops, num)
    #print(rays_radii[100]-rays_radii[1])

    import colorcet
    fig, ax = plt.subplots(1,1)
    plt.rcParams['text.usetex'] = True
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['figure.figsize'] = [6, 4]
    plt.rcParams['axes.facecolor']= 'whitesmoke'
    
    T_plot = np.log10(rays_T)
    T_plot = np.nan_to_num(T_plot, neginf= -19)
    radii_toplot = []
    for j in range(len(rays_radii)):
        radius = np.delete(rays_radii[j], -1)
        radius /= Rsol_to_cm
        radii_toplot.append(radius)

    # ax.set_ylabel('Observers', fontsize = 14)
    # ax.set_xlabel(r'r [R$_\odot$]', fontsize = 14)
    # img = ax.pcolormesh(radii_toplot, range(len(rays_T)), T_plot, cmap = 'cet_fire')
    #                     #vmin = -17, vmax = - 7)
    # cb = plt.colorbar(img)
    # cb.set_label(r'Density [g/cm$^3$]', fontsize = 14)
    #ax.set_title('N: ' + str(num), fontsize = 16)

    # plt.show()
    plt.plot(radii_toplot[80], rays_T[80])
    plt.loglog()
    plt.xlim(0.56,3e4)
    #plt.plot(radii_toplot[20], rays_T[20])
    plt.show()
    