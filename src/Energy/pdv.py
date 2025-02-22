#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 15:33:35 2025

@author: konstantinos
"""
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import KDTree
from scipy.interpolate import griddata
from tqdm import tqdm
import matlab.engine
eng = matlab.engine.start_matlab()

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
X, Y, Z, Den, T, Rad, Vol, divV, P, day = local_loader(6, 350, 'PdV')
Rad_den = Rad * Den * c.en_den_converter # [erg/cm3]

def tree_subsample(X, Y, Z, Den, T, Rad, Vol, divV, P, num = 100):
    x_start = -5*amin
    x_stop =  5*amin
    y_start = -5*amin
    y_stop = 5*amin
    z_start = -5*amin
    z_stop = 5*amin
        
    xs = np.linspace(x_start, x_stop, num)
    ys = np.linspace(y_start, y_stop, num)
    zs = np.linspace(z_start, z_stop, num)
    
    sim_value = [X, Y, Z] 
    sim_value = np.transpose(sim_value) #array of dim (number_points, 3)
    sim_tree = KDTree(sim_value) 
    
    gridded_indexes =  np.zeros(( len(xs), len(ys), len(zs) ))
    rad_g =  np.zeros(( len(xs), len(ys), len(zs) ))
    Den_g =  np.zeros_like(rad_g)
    T_g =  np.zeros_like(rad_g)
    vol_g =  np.zeros_like(rad_g)
    X_g =  np.zeros_like(rad_g)
    Y_g =  np.zeros_like(rad_g)
    Z_g =  np.zeros_like(rad_g)
    divV_g =  np.zeros_like(rad_g)
    P_g =  np.zeros_like(rad_g)
    for i in tqdm(range(len(xs))):
        for j in range(len(ys)):
            for k in range(len(zs)):
                queried_value = [xs[i], ys[j], zs[k]]
                _, idx = sim_tree.query(queried_value)
                                    
                # Store
                Den_g[i, j, k] = Den[idx]
                T_g[i, j, k] = T[idx]
                rad_g[i, j, k] = Rad[idx]
                vol_g[i, j, k] = Vol[idx]
                X_g[i, j, k] = X[idx]
                Y_g[i, j, k] = Y[idx]
                Z_g[i, j, k] = Z[idx]
                divV_g[i, j, k] = divV[idx]
                P_g[i, j, k] = P[idx]

    return X_g, Y_g, Z_g, Den_g, T_g, rad_g, vol_g, divV_g, P_g
X_g, Y_g, Z_g, den_g, t_g, rad_g, vol_g, divV_g, P_g = tree_subsample(X, Y, Z, 
                                                                     Den, T, 
                                                                     Rad_den, 
                                                                     Vol,
                                                                     divV, P)
            
sigma_rossland = eng.interp2(T_opac_ex, Rho_opac_ex, rossland_ex.T, 
                             np.log(t_g), np.log(den_g), 'linear', 0)
sigma_rossland = np.array(sigma_rossland)[0]
underflow_mask = sigma_rossland != 0.0
X_g, Y_g, Z_g, den_g, t_g, rad_g, vol_g, sigma_rossland = masker(underflow_mask, 
[X_g, Y_g, Z_g, den_g, t_g, rad_g, vol_g, sigma_rossland])
sigma_rossland_eval = np.exp(sigma_rossland)

# Summon the Elad-ness
#%% Cell radius
dx = 0.5 * vol_g**(1/3)
    
# Get the Grads    
f_inter_input = np.array([ X, Y, Z ]).T

gradx_p = griddata( f_inter_input, Rad_den, method = 'linear',
                    xi = np.array([ X_g+dx, Y_g, Z_g]).T )
gradx_m = griddata( f_inter_input, Rad_den, method = 'linear',
                    xi = np.array([ X_g-dx, Y_g, Z_g]).T )
gradx = (gradx_p - gradx_m)/ (2*dx)
gradx = np.nan_to_num(gradx, nan =  0)

grady_p = griddata( f_inter_input, Rad_den, method = 'linear',
                    xi = np.array([ X_g, Y_g+dx, Z_g]).T )
grady_m = griddata( f_inter_input, Rad_den, method = 'linear',
                    xi = np.array([ X_g, Y_g-dx, Z_g]).T )
grady = (grady_p - grady_m)/ (2*dx)
grady = np.nan_to_num(grady, nan =  0)

gradz_p = griddata( f_inter_input, Rad_den, method = 'linear',
                    xi = np.array([ X_g, Y_g, Z_g+dx]).T )
gradz_m = griddata( f_inter_input, Rad_den, method = 'linear',
                    xi = np.array([ X_g, Y_g, Z_g-dx]).T )
gradz_m = np.nan_to_num(gradz_m, nan =  0)
gradz = (gradz_p - gradz_m)/ (2*dx)

grad = np.sqrt(gradx**2 + grady**2 + gradz**2)

# Krumholz
R_lamda = np.abs(grad) / (sigma_rossland_eval * Rad_den)
R_lamda[R_lamda < 1e-10] = 1e-10
lamda = 3 * (1/np.tanh(R_lamda) - 1/R_lamda) / R_lamda
R2 = lamda + lamda**2 * R_lamda
P_rad = Rad_den/2 * (1 - R2) @ np.eye(len(R2))

# DO IT
PdV = np.sum((P_rad + P) * divV)
