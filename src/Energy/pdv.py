#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 15:33:35 2025

@author: konstantinos
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import numba

from scipy.spatial import KDTree
from scipy.interpolate import griddata
from tqdm import tqdm
import matlab.engine
eng = matlab.engine.start_matlab()
from lmfit import Model

import src.Utilities.prelude as c 
from src.Utilities.loaders import local_loader

from src.Opacity.LTE_loader import T_opac_ex, Rho_opac_ex, rossland_ex
# Opacity Input
T_cool2 = T_opac_ex
Rho_cool2 = Rho_opac_ex
rossland2 = rossland_ex

def masker(mask, list_of_quantities):
    new_list = []
    for quantity in list_of_quantities:
        new_list.append(quantity[mask])
    return (*new_list,)

mstar = 0.5
rstar = 0.47
m = 6
Rt = rstar * (10**m/mstar)**(1/3)
amin = Rt * (10**m/mstar)**(1/3)
fix = 350
subsample = 10
X, Y, Z, Den, T, Rad, Vol, divV, P, day = local_loader(6, 350, 'PdV', subsample)

# convert EVERYTHING to cgs
X *= c.Rsol_to_cm
Y *= c.Rsol_to_cm
Z *= c.Rsol_to_cm
Rad_den = Rad * Den * c.en_den_converter # [erg/cm3]
Den *= c.den_converter
divV /= c.t
P *= c.en_den_converter

# Opacity
sigma_rossland = eng.interp2(T_opac_ex, Rho_opac_ex, rossland_ex.T, 
                             np.log(T), np.log(Den), 'linear', 0)
sigma_rossland = np.array(sigma_rossland)[0]
underflow_mask = sigma_rossland != 0.0
X, Y, Z, Den, T, Rad_den, Vol, divV, P, sigma_rossland = masker(underflow_mask, 
[X, Y, Z, Den, T, Rad_den, Vol, divV, P, sigma_rossland])
sigma_rossland_eval = np.exp(sigma_rossland)

def neighbourhood_builder(X, Y, Z):
    sim_value = [X, Y, Z] 
    sim_value = np.transpose(sim_value) #array of dim (number_points, 3)
    sim_tree = KDTree(sim_value) 
    
    neighbours = []
    for cell in range(len(X)):
        _, idxs = sim_tree.query([X[cell], Y[cell], Z[cell]], k = 5)
        neighbours.append(idxs)
    return neighbours

def gradient_calc_mine(X, Y, Z, Rad_den, neighbours):
    grad = np.zeros_like(X)
    for cell in tqdm(range(len(X))):
        these_neighbours = neighbours[cell]
        these_erads = np.zeros_like(these_neighbours)
        these_xs = np.zeros_like(these_neighbours)
        these_ys = np.zeros_like(these_neighbours)
        these_zs = np.zeros_like(these_neighbours)
    
        for nidx, neighbour in enumerate(these_neighbours):
            these_erads[nidx] = Rad_den[neighbour]
            these_xs[nidx] = X[neighbour]
            these_ys[nidx] = Y[neighbour]
            these_zs[nidx] = Z[neighbour]
            
        sortx = np.argsort(these_xs)
        gradx = np.gradient(these_erads[sortx], these_xs[sortx])
        gradx = np.nan_to_num(gradx, nan =  0)
    
        sorty = np.argsort(these_ys)
        grady = np.gradient(these_erads[sorty], these_ys[sorty])
        grady = np.nan_to_num(grady, nan =  0)
    
        
        sortz = np.argsort(these_zs)
        gradz = np.gradient(these_erads[sortz], these_zs[sortz])
        gradz = np.nan_to_num(gradz, nan =  0)
       
        grad[cell] = np.mean(np.sqrt(gradx**2 + grady**2 + gradz**2))
    return grad

#%% Summon the Elad-ness
def elad_model(deltax, gradx, grady, gradz, Ecm):
    'grad, deltax are 3 vectors, Ecm is a scalar'
    grad = np.array([gradx, grady, gradz])
    return Ecm + np.dot(grad, deltax.T)

def gradient_lsq(X, Y, Z, Rad_den, neighbours):
    grad = np.zeros_like(X)
    for cell in tqdm(range(len(X))):
        Ecm_this = Rad_den[cell]
        these_neighbours = neighbours[cell]
        Ens = np.zeros_like(these_neighbours)
        
        these_deltas = np.zeros((len(these_neighbours),3))
        pos_cell = np.array([X[cell], Y[cell], Z[cell]])
        for nidx, neighbour in enumerate(these_neighbours):
            Ens[nidx] = Rad_den[neighbour]

            pos_neigh = np.array([X[neighbour], Y[neighbour], Z[neighbour]])
            delta_pos = pos_cell - pos_neigh
            these_deltas[nidx] = pos_cell
    
        fmodel = Model(elad_model)
        params = fmodel.make_params(gradx = 1, grady = 1, gradz = 1, 
                                    Ecm=Ecm_this)
        params['Ecm'].vary = False
        params['Ecm'].value = Ecm_this
        
        lsq = fmodel.fit(Ens, params, deltax = these_deltas)
        gradx = lsq.params['gradx'].value
        grady = lsq.params['grady'].value
        gradz = lsq.params['gradz'].value
        
        grad[cell] = gradx**2 + grady**2 + gradz**2
    return grad
neighbours = neighbourhood_builder(X, Y, Z)
# grad = gradient_lsq(X, Y, Z, Rad_den, neighbours)
grad = gradient_calc_mine(X, Y, Z, Rad_den, neighbours)
    
#%% Krumholz
R_lamda = np.abs(grad) / (sigma_rossland_eval * Rad_den)
R_lamda[R_lamda < 1e-10] = 1e-10
lamda = (1/np.tanh(R_lamda) - 1/R_lamda) / R_lamda
R2 = lamda + lamda**2 * R_lamda
P_rad = Rad_den/2 * (1 - R2) * np.eye(len(R2))
P_rad = np.nan_to_num(P_rad)
P_rad = np.diag(P_rad)

#%% Plot
fig, ax = plt.subplots(1,1, figsize = (3,3))

densort = np.argsort(Den)
posmask = P_rad > 0
negmask = P_rad < 0

ax.plot(Den[densort][posmask], P_rad[densort][posmask], c = 'k', ls = '', 
         marker = 'o', markersize = 1)
ax.plot(Den[densort][negmask], P_rad[densort][negmask], c = c.AEK, ls = '', 
         marker = 'o', markersize = 1)
ax.plot(Den[densort], P[densort], c = 'maroon', ls = '', 
         marker = 'o', markersize = 1)

# Pretty
ax.set_xlabel('Density [g/cm$^3$]')
ax.set_ylabel('Pressure [erg/cm$^3$]')
# ax.set_ylabel('Flux Limiter $\lambda$')
ax.set_yscale('log')
ax.set_xscale('log')

ax.plot([], [], c = 'k', ls = '', 
         marker = 'o', markersize = 3, label = '$P_\mathrm{rad}>0$')
ax.plot([], [], c = c.AEK, ls = '', 
         marker = 'o', markersize = 3, label = '$P_\mathrm{rad}<0$')
ax.plot([], [], c = 'maroon', ls = '', 
         marker = 'o', markersize = 3, label = '$P_\mathrm{gas}$')

ax.legend(frameon = 0)


#%%
# DO IT
dV = divV * Vol
PdV = np.sum((P_rad + P) * dV) * subsample # * c.power_converter
print(f'\n PdV: {PdV:.2e} erg/s')
