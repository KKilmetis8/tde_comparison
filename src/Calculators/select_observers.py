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
    #xyz = [x, y, z]
    return x,y,z

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
    wanted_phis = [0, np.pi, np.pi/2, 3*np.pi/2, 0, 0]
    x_wanted = np.zeros(len(wanted_thetas))
    y_wanted = np.zeros(len(wanted_thetas))
    z_wanted = np.zeros(len(wanted_thetas)) 
 
    for i in range(len(wanted_thetas)):
        x_wanted[i], y_wanted[i], z_wanted[i] = find_sph_coord(wanted_thetas[i], wanted_phis[i])

    # for idx in range(len(wanted_thetas)):
    #     wanted_theta = wanted_thetas[idx]
    #     wanted_phi = wanted_phis[idx]
    #     wanted_index = select_observer(wanted_theta, wanted_phi, thetas, phis)
    #     print('index ',wanted_index)

    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    # Create a sphere
    r = 0.5
    phi_sph, theta_sph = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
    x = r*np.sin(phi_sph)*np.cos(theta_sph)
    y = r*np.sin(phi_sph)*np.sin(theta_sph)
    z = r*np.cos(phi_sph)
    ax.plot_surface(x, y, z, color='orange', alpha = 0.5)

    #Set arrow for axis
    xar = np.zeros(len(x_wanted))
    yar = np.zeros(len(x_wanted))
    zar = np.zeros(len(x_wanted))
    # ax.quiver(xar,yar,zar, dx, dy, dz, arrow_length_ratio=0.1, color = 'k')
    col = ['b', 'r', 'k', 'lime', 'magenta', 'aqua']
    ax.quiver(xar, yar, zar, x_wanted, y_wanted, z_wanted, arrow_length_ratio=0.1, color = 'k')

    ax.scatter(x_wanted,y_wanted,z_wanted, color = col, s=20)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.tight_layout()
    plt.savefig('Final plot/observerspectra.png')
    plt.show()