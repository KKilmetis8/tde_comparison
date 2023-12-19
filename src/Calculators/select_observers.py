"""
Created on Mon Dec 11 2023

@author: paola

"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [5 , 4]
NSIDE = 4


def select_observer(wanted_theta, wanted_phi, observers):
    """ Gives thetas, phis from helpix and 
    the index of the points closer to the one given by (wanted_theta, wanted_phi)"""
    thetas = np.zeros(192)
    phis = np.zeros(192)
    for iobs in range(len(observers)): 
        thetas[iobs] = observers[iobs][0]
        phis[iobs] =  observers[iobs][1]
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

    return thetas, phis, index

if __name__ == '__main__':
    wanted_theta = 0
    wanted_phi = 0
  
    thetas, phis, index = select_observer(wanted_theta, wanted_phi)
    fig, ax = plt.subplots(1,1, subplot_kw=dict(projection="mollweide"))
    ax.scatter(phis, thetas, c = 'k', s=20, marker = 'h')
    ax.scatter(phis[index], thetas[index], c = 'r', s=20, marker = 'h')
    plt.grid(True)
    plt.show()