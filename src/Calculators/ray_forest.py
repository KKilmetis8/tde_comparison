#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gives ray which are logspaced. Around photosphere they sould be 1.
Created on Tue Oct 10 10:19:34 2023

@author: paola, konstantinos

"""

# Vanilla
import h5py
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# Chocolate
import src.Utilities.prelude as c
from src.Utilities.parser import parse
from src.Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    pre = '/home/kilmetisk/data1/TDE/'

import src.Utilities.selectors as s
#%% Constants & Converter

def find_sph_coord(r, theta, phi):
    x = r * np.sin(np.pi-theta) * np.cos(phi) #Elad has just theta
    y = r * np.sin(np.pi-theta) * np.sin(phi)
    z = r * np.cos(np.pi-theta)
    return [x,y,z]

# This just packs rays
class ray_keeper:
    def __init__(self, tree_indexes, rays_T, rays_den, rays_rad, rays_ie, 
               rays_radii, rays_vol, rays_v):
        self.tree_indexes = tree_indexes
        self.T = rays_T
        self.den = rays_den
        self.rad = rays_rad
        self.ie = rays_ie
        self.radii = rays_radii
        self.vol = rays_vol
        self.v = rays_v

def ray_maker_forest(fix, m, star, check, thetas, phis, stops, num, opacity, beta,
                     starflag = False): 
    """ 
    Num is 1001 because for blue we then delete the last cell.
    Outputs are in CGS with exception of ray_vol (in solar units).
    """
    fix = str(fix)
    
    if alice:
        args = parse()
        sim = args.name
        mstar = args.mass
        rstar = args.radius
        Mbh = args.blackhole
    else:
        rstar = 0.47
        mstar = 0.5
        Mbh = 10**(int(m))
        pre = ''
        sim = f'{m}'
    Rt = rstar * (Mbh/mstar)**(1/3) 
    apocenter = rstar * (Mbh/mstar)**(1/3)
    
    # Load data
    X = np.load(f'{pre}{sim}/{fix}/CMx_{fix}.npy') 
    Y = np.load(f'{pre}{sim}/{fix}/CMy_{fix}.npy')
    Z = np.load(f'{pre}{sim}/{fix}/CMz_{fix}.npy')
    VX = np.load(f'{pre}{sim}/{fix}/Vx_{fix}.npy') 
    VY = np.load(f'{pre}{sim}/{fix}/Vy_{fix}.npy')
    VZ = np.load(f'{pre}{sim}/{fix}/Vz_{fix}.npy')
    T = np.load(f'{pre}{sim}/{fix}/T_{fix}.npy') 
    Den = np.load(f'{pre}{sim}/{fix}/Den_{fix}.npy')
    Rad = np.load(f'{pre}{sim}/{fix}/Rad_{fix}.npy')
    IE = np.load(f'{pre}{sim}/{fix}/IE_{fix}.npy') 
    Vol = np.load(f'{pre}{sim}/{fix}/Vol_{fix}.npy')

    if starflag:
        Star = np.load(pre + fix + '/Star_' + fix + '.npy')
    if opacity == 'cloudy': # elad 
        Tcool_min = np.loadtxt('src/Opacity/cloudy_data/Tcool_ext.txt')[0]
    
    # Convert Energy / Mass to Energy Density in CGS
    Rad *= Den 
    Rad *= c.en_den_converter
    IE *= Den 
    IE *= c.en_den_converter
    Den *= c.den_converter 
    
    # make a tree
    sim_value = [X, Y, Z] 
    sim_value = np.transpose(sim_value) #array of dim (number_points, 3)
    sim_tree = KDTree(sim_value) 

    # Ensure that the regular grid cells are smaller than simulation cells
    # NOTE: for the 10^6 what do we do for the other???
    # We are ignoring beta here. -> not anymore
    start = Rt/200 /beta # 1e-0.25 Solar radii  (arbitrary choice ELAD made)
    rays_radii = []
    
    # you take num-1 beacause in blue you will delete the last cell of radii
    tree_indexes = np.zeros((len(thetas), num-1))
    rays_T = np.zeros((len(thetas), num-1))
    rays_den = np.zeros((len(thetas), num-1))
    rays_rad = np.zeros((len(thetas), num-1))
    rays_ie = np.zeros((len(thetas), num-1))
    rays_vol = np.zeros((len(thetas), num-1))
    rays_v = np.zeros((len(thetas), num-1))

    for j in range(len(thetas)):
        stop =  stops[j]
        log_start = np.log10(start)
        log_stop = np.log10(stop)
        log_radii = np.linspace(log_start, log_stop, num) #simulator units
        radii = 10**log_radii
        
        for k in range(num-1):
            radius = radii[k]
            queried_value = find_sph_coord(radius, thetas[j], phis[j])
            # queried_value[0] += Rt #if you don't do -Rt before. Thus you consider the pericentre as origin
            _, idx = sim_tree.query(queried_value)

            # Store
            tree_indexes[j][k] = idx 
            if opacity == 'cloudy': # elad
                rays_T[j][k] = max(T[idx], Tcool_min)
            else:
                rays_T[j][k] = T[idx]

            # Apply star mask, throw away the fluff
            if starflag:
                cell_star = Star[idx]
                if opacity == 'cloudy':
                    if ((1-cell_star) > 1e-3):
                        rays_den[j][k] = 0
                    else:
                        rays_den[j][k] = Den[idx] 
                else:
                    if cell_star < 0.5:
                        rays_den[j][k] = 0
                    else:
                        rays_den[j][k] = Den[idx]
            else:
                rays_den[j][k] = Den[idx]
                
            # Apply mask
            rays_rad[j][k] = Rad[idx] 
            rays_ie[j][k] = IE[idx] 
            rays_vol[j][k] = Vol[idx] # not in CGS
            vel = np.sqrt(VX[idx]**2 + VY[idx]**2 + VZ[idx]**2)
            vel *= c.Rsol_to_cm / c.t # convert in CGS
            rays_v[j][k] = vel

        # Convert to CGS
        radii *= c.Rsol_to_cm
        rays_radii.append(radii)
    
    # Remove Bullshit
    rays_T = np.nan_to_num(rays_T, neginf = 0)
    rays_den = np.nan_to_num(rays_den, neginf = 0)
    rays_rad = np.nan_to_num(rays_rad, neginf = 0)
    rays_ie = np.nan_to_num(rays_ie, neginf = 0)
    
    rays_local = ray_keeper(tree_indexes, rays_T, rays_den, rays_rad, rays_ie, 
                            rays_radii, rays_vol, rays_v)
    
    return rays_local

def ray_finder(filename):
    # Get simulation box
    box = np.zeros(6)
    with h5py.File(filename, 'r') as fileh:
        for i in range(len(box)):
            box[i] = fileh['Box'][i]
            
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
        observers.append( (theta, phi) )
        xyz = find_sph_coord(1, theta, phi) # r=1 to be on the unit sphere
        xyz_grid.append(xyz)
        mu_x = xyz[0]
        mu_y = xyz[1]
        mu_z = xyz[2]
                
        # Box is for dynamic ray making
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
        
    return thetas, phis, stops, xyz_grid
 
if __name__ == '__main__':
    m = 6
    snap = 882
    check = 'fid'
    num = 1000
    filename = f"{m}/{snap}/snap_{snap}.h5"
    
    opacity_kind = s.select_opacity(m)
        
    # Get thetas, phis and where each ray stops
    thetas, phis, stops, xyz_grid = ray_finder(filename)
    rays = ray_maker_forest(snap, m, check, thetas, phis, stops, num, opacity_kind)

    T_plot = np.log10(rays.T)
    T_plot = np.nan_to_num(T_plot, neginf= -19)
    radii_toplot = []
    for j in range(len(rays.radii)):
        radius = np.delete(rays.radii[j], -1)
        radius /= c.Rsol_to_cm
        radii_toplot.append(radius)

    # ax.set_ylabel('Observers', fontsize = 14)
    # ax.set_xlabel(r'r [R$_\odot$]', fontsize = 14)
    # img = ax.pcolormesh(radii_toplot, range(len(rays.T)), T_plot, cmap = 'cet_fire')
    #                     #vmin = -17, vmax = - 7)
    # cb = plt.colorbar(img)
    # cb.set_label(r'Density [g/cm$^3$]', fontsize = 14)
    #ax.set_title('N: ' + str(num), fontsize = 16)

    # plt.show()
    plt.plot(radii_toplot[80], rays.T[80])
    plt.loglog()
    plt.xlim(0.56,3e4)
    plt.show()
    