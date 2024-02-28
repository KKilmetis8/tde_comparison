"""
Created on Mon Dec 11 2023

@author: paola

Select a spectra among the ones given by healpix

"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

from src.Calculators.ray_tree import ray_maker
import healpy as hp
import colorcet

import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [5 , 4]

def find_sph_coord(theta,phi):
    x = np.sin(np.pi-theta) * np.cos(phi) #because theta starts from the z axis: we're flipped
    y = np.sin(np.pi-theta) * np.sin(phi)
    z = np.cos(np.pi-theta)
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

def old_select_observer(wanted_theta, wanted_phi, thetas, phis):
    """ Gives thetas, phis from helpix and 
    the index of the points closer to the one given by (wanted_theta, wanted_phi)"""
    dist = np.zeros(192)
    for i in range(len(thetas)): 
        delta_theta = wanted_theta -  thetas[i]
        delta_phi = wanted_phi -  phis[i]
        # Haversine formula
        arg = np.sin(delta_theta / 2)**2 + np.cos(wanted_theta) * np.cos(thetas[i]) * np.sin(delta_phi/2)**2
        dist[i] = 2 * np.arctan2( np.sqrt(arg), np.sqrt(1-arg))
    
    index = np.argmin(np.abs(dist))
    # index = np.where(np.abs(dist) == dist.min())

    return index

if __name__ == '__main__':
    m = 6
    snap = 844
    check = 'fid'
    _, observers, _, _, _, _, _, _, _ = ray_maker(snap, m, check, 500)
    plot = '3dim'

    # Observers from ray_maker: theta in [0,pi], phi in [0,2pi]
    thetas = np.zeros(192)
    phis = np.zeros(192) 
    x_healpix = np.zeros(192)
    y_healpix = np.zeros(192)
    z_healpix = np.zeros(192)
    for iobs in range(len(observers)): 
        thetas[iobs] = observers[iobs][0]
        phis[iobs] =  observers[iobs][1]
        x_healpix[iobs], y_healpix[iobs], z_healpix[iobs] = find_sph_coord(thetas[iobs], phis[iobs])
    

    if plot == '3dim':
        # Observer you would like to see
        wanted_thetas = [np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi, 0] # x, -x, y, -y, z, -z
        wanted_phis = [0, np.pi, np.pi/2, 3*np.pi/2, 0, 0]
        x_wanted = np.zeros(len(wanted_thetas))
        y_wanted = np.zeros(len(wanted_thetas))
        z_wanted = np.zeros(len(wanted_thetas)) 
        for i in range(len(wanted_thetas)):
            x_wanted[i], y_wanted[i], z_wanted[i] = find_sph_coord(wanted_thetas[i], wanted_phis[i])
        
        # Plot the ones chosen 
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
        plt.savefig('Final_plot/observerspectra.png')
        plt.title('Real')

        # Plot healpix with axis 
        fig = plt.figure()
        ax = fig.add_subplot(projection = '3d')
        ax.scatter(x_healpix, y_healpix, z_healpix, color = 'b', s=10)
        ax.quiver(xar, yar, zar, x_wanted, y_wanted, z_wanted, arrow_length_ratio=0.1, color = 'k')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Observers from healpix nearest to the ones you want
        x_selected = np.zeros(len(wanted_thetas))
        x_selected2 = np.zeros(len(wanted_thetas))

        y_selected = np.zeros(len(wanted_thetas))
        y_selected2 = np.zeros(len(wanted_thetas))

        z_selected = np.zeros(len(wanted_thetas))
        z_selected2 = np.zeros(len(wanted_thetas))

        for idx in range(len(wanted_thetas)):
            wanted_theta = wanted_thetas[idx]
            wanted_phi = wanted_phis[idx]
            wanted_index = old_select_observer(wanted_theta, wanted_phi, thetas, phis)
            wanted_index_elad = select_observer(wanted_theta, wanted_phi, thetas, phis)
            x_selected[idx], y_selected[idx], z_selected[idx] = find_sph_coord(thetas[wanted_index], phis[wanted_index])
            x_selected2[idx], y_selected2[idx], z_selected2[idx] = find_sph_coord(thetas[wanted_index_elad], phis[wanted_index_elad])


        print('X:' , x_selected)
        print('Y:' , y_selected)
        print('Z:' , z_selected)
        
        # What Elad Does
        xhat = (1,0,0)
        observers = np.array([x_healpix,
                             y_healpix, 
                             z_healpix]).T
        inner_product = [np.dot(xhat, observer) for observer in observers]
        blue_dot_idx = np.argmax(inner_product)
        
        # Plotting
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

        ax.scatter(x_selected,y_selected,z_selected, color = col, s=20)
        ax.scatter(x_selected2,y_selected2,z_selected2, color = col, s=50, marker = 's')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.tight_layout()
        plt.savefig('Final_plot/observerspectra_healp.png')
        plt.title('dot vs haversine')
        
        # # Elad's point
        # ax.scatter(observers[blue_dot_idx][0], observers[blue_dot_idx][1], 
        #             observers[blue_dot_idx][2],
        #             c = 'b', marker = 's', s = 50, alpha = 0.5)
        # # Elad's point
        # ax.scatter(observers[104][0], observers[104][1], 
        #             observers[104][2],
        #             c = 'b', marker = 's', s = 50, alpha = 0.5)
        # ax.scatter(observers[88][0], observers[88][1], 
        #            observers[88][2],
        #            c = 'r', marker = 's', s = 50, alpha = 0.5)
        
    if plot == '2dim':
        Mbh = 10**m 
        Rt =  Mbh**(1/3) # Msol = 1, Rsol = 1
        apocenter = 2 * Rt * Mbh**(1/3) 
        pre = f'data/denproj/{m}'#/{m}-{check}'
        sim = f'{m}-{check}'
        data = np.loadtxt(f'{pre}/denproj{sim}{snap}.txt')
        x_radii = np.loadtxt(pre + '/xarray' + sim + '.txt') #simulator units
        y_radii = np.loadtxt(pre + '/yarray' + sim + '.txt') #simulator units

        # Observer you would like to see
        wanted_thetas = [np.pi/2, np.pi/2, np.pi/2, np.pi/2] # x, -x, y, -y, z, -z
        wanted_phis = [0, np.pi, np.pi/2, 3*np.pi/2]
        x_wanted = np.zeros(len(wanted_thetas))
        y_wanted = np.zeros(len(wanted_thetas))
        z_wanted = np.zeros(len(wanted_thetas)) 
        for i in range(len(wanted_thetas)):
            x_wanted[i], y_wanted[i], z_wanted[i] = find_sph_coord(wanted_thetas[i], wanted_phis[i])
            # Need the following for plotting
            # if y_wanted[i] > 0.2:
            #     y_wanted[i] = 0.15
            # if y_wanted[i] < -0.2:
            #     y_wanted[i] = -0.15
            # if x_wanted[i] > 0.5:
            #     x_wanted[i] = max(x_radii)/apocenter -0.005
            # if x_wanted[i] < -0.5:
            #     x_wanted[i] = -0.95
        col = ['b', 'r', 'k', 'lime']

        # Observers from healpix nearest to the ones you want
        x_selected = np.zeros(len(wanted_thetas))
        y_selected = np.zeros(len(wanted_thetas))
        z_selected = np.zeros(len(wanted_thetas))
        for idx in range(len(wanted_thetas)):
            wanted_theta = wanted_thetas[idx]
            wanted_phi = wanted_phis[idx]
            wanted_index = select_observer(wanted_theta, wanted_phi, thetas, phis)
            x_selected[idx], y_selected[idx], z_selected[idx] = find_sph_coord(thetas[wanted_index], phis[wanted_index])

        fig, ax = plt.subplots(1,1)
        den_plot = np.nan_to_num(data, nan = -1, neginf = -1)
        den_plot = np.log10(den_plot)
        den_plot = np.nan_to_num(den_plot, neginf= 0)

        ax.set_xlabel(r' X [$x/R_a$]', fontsize = 14)
        ax.set_ylabel(r' Y [$y/R_a$]', fontsize = 14)
        img = ax.pcolormesh(x_radii/apocenter, y_radii/apocenter, den_plot.T, 
                            cmap = 'cet_fire', alpha = 0.8,
                            vmin = 0, vmax = 6)
        ax.scatter(x_wanted, y_wanted, c = col, marker = 'X')
        ax.scatter(x_selected, y_selected, c = col)
        ax.scatter(x_healpix[88:103], y_healpix[88:103], s = 8)
        plt.savefig('Final_plot/observers_inproj.png')

        plt.show()
        

