"""
Created on Mon Dec 11 2023

@author: paola

Select a spectra among the ones given by healpix

"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

from src.Calculators.ray_tree import ray_maker
import healpy as hp

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [5 , 4]
NSIDE = 4

def find_sph_coord(theta,phi):
    x = np.sin(np.pi-theta) * np.cos(phi) #because theta starts from the z axis: we're flipped
    y = np.sin(np.pi-theta) * np.sin(phi)
    z = np.cos(np.pi-theta)
    xyz = [x, y, z]
    return xyz

def select_observer(wanted_theta, wanted_phi, thetas, phis):
    """ Gives thetas, phis from helpix and 
    the index of the points closer to the one given by (wanted_theta, wanted_phi)"""

    dist = np.zeros(192)
    for i in range(192):
        delta_theta = wanted_theta -  thetas[i]
        delta_phi = wanted_phi -  phis[i]
        # Haversine formula
        arg = np.sin(delta_theta / 2)**2 + np.cos(wanted_theta) * np.cos(thetas[i]) * np.sin(delta_phi/2)**2
        dist[i] = 2 * np.arctan2( np.sqrt(arg), np.sqrt(1-arg))

    index = np.where(np.abs(dist) == dist.min()) # not argmin since it gives only 1 point
    index = np.concatenate(index)
    index = index[0] #ORRIBLE

    return index

if __name__ == '__main__':
    # _, observers, _, _, _, _, _ = ray_maker(844, 6, 'fid')

    # # angles from ray_maker: theta in [0,pi], phi in [0,2pi]
    # thetas = np.zeros(192)
    # phis = np.zeros(192) 
    # for iobs in range(len(observers)): 
    #     thetas[iobs] = observers[iobs][0]
    #     phis[iobs] =  observers[iobs][1]
    
    wanted_thetas = [np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi, 0] # x, -x, y, -y, z, -z
    wanted_phis = [0, np.pi/2, np.pi, 3*np.pi/2, 0, 0]
    xyz_wanted = [] #np.zeros(len(wanted_thetas))
    for i in range(len(wanted_thetas)):
        xyz_wanted.append(find_sph_coord(wanted_thetas[i], wanted_phis[i]))

    # for idx in range(len(wanted_thetas)):
    #     wanted_theta = wanted_thetas[idx]
    #     wanted_phi = wanted_phis[idx]
    #     wanted_index = select_observer(wanted_theta, wanted_phi, thetas, phis)
    #     print('index ',wanted_index)

    # Create a sphere
    # r = 1
    # phi_sph, theta_sph = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
    # x = r*np.sin(phi_sph)*np.cos(theta_sph)
    # y = r*np.sin(phi_sph)*np.sin(theta_sph)
    # z = r*np.cos(phi_sph)

    # #Set colours and render
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # yy, zz = np.meshgrid(range(-1,2), range(-1,2))
    # xx = yy*0
    # zz2, xx2 = np.meshgrid(range(-1,2), range(-1,2))
    # yy2 = xx2*0
    # xx3, yy3 = np.meshgrid(range(-1,2), range(-1,2))
    # zz3 = yy3*0
    # ax.plot_surface(xx, yy, zz, alpha = 0.6, color = 'b')
    # ax.plot_surface(xx2, yy2, zz2, alpha = 0.6, color = 'r')
    # ax.plot_surface(xx3, yy3, zz3, alpha = 0.6, color = 'green')
    # ax.plot_surface(x, y, z, color='c')
    # for i in range(len(xyz_wanted)):
    #     ax.scatter(xyz_wanted[i][0],xyz_wanted[i][1],xyz_wanted[i][2],color="k",s=20)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_xlim([-1,1])
    # ax.set_ylim([-1,1])
    # ax.set_zlim([-1,1])
    # plt.tight_layout()
    # plt.show()

    # thetas_toplot = thetas - np.pi/2 #Enforce theta in -pi/2 to pi/2 to plot
    # phis_toplot = phis-np.pi # Enforce theta in -pi to pi to plot
    # fig, ax = plt.subplots(1,1, subplot_kw=dict(projection="mollweide"))
    # ax.scatter(phis_toplot, thetas_toplot, c = 'k', s=20, marker = 'h')
    # ax.scatter(phis_toplot[index], thetas_toplot[index], c = 'r', s=20, marker = 'h')
    # plt.grid(True)
    # plt.show()