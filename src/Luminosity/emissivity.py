"""
Created on January 2024

@author: paola 

Calculate (emissivity *e^-\tau).

"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

from src.Utilities.isalice import isalice
alice, plot = isalice()

# Vanilla imports
import numpy as np
import matplotlib.pyplot as plt

# Chocolate Imports
from src.Opacity.cloudy_opacity import old_opacity 

# Constants
c = 2.99792458e10 #[cm/s]
h = 6.62607015e-27 #[gcm^2/s]
Kb = 1.380649e-16 #[gcm^2/s^2K]
alpha = 7.5646 * 10**(-15) # radiation density [erg/cm^3K^4]
Rsol_to_cm = 6.957e10

def emissivity(Temperature: float, Density: float, tau: float, radius: float, dr: float):
    """ Compute (emissivity *e^-\tau) for every cell."""
    Tmax = np.power(10,8) 
    if Temperature > Tmax:
        # Scale as Kramers the last point 
        kplanck_0 = old_opacity(Tmax, Density, 'planck') 
        k_planck = kplanck_0 * (Temperature/Tmax)**(-3.5)
    else:
        k_planck = old_opacity(Temperature, Density, 'planck') 
    ecool = 4 * np.pi * dr * radius**3 * alpha * c * Temperature**4 * k_planck * np.exp(-tau) ## do hve I to multiply for 192??
    return ecool

def ray_emissivity(rays_T, rays_den, rays_cumulative_taus, radii, dr):
    """ Compute (emissivity *e^-\tau) for every cell."""
    rays_ecool = []
    for j in range(len(rays_T)):
        ecool = np.zeros(len(rays_cumulative_taus[j]))

        for i in range(len(rays_cumulative_taus[j])):        
            # Temperature, Density and volume: np.array from near to the BH to far away. 
            # Thus we will use negative index in the for loop.
            # tau: np.array from outside to inside.
            reverse_idx = -i -1
            radius = radii[reverse_idx]
            T = rays_T[j][reverse_idx]
            rho = rays_den[j][reverse_idx] 
            opt_depth = rays_cumulative_taus[j][i]
            ecool[i] = emissivity(T, rho, opt_depth, radius, dr)

        rays_ecool.append(ecool)

    return rays_ecool

def find_threshold_temp(rays_T, rays_den, rays_cumulative_taus, radii):
    dr = 2 * (radii[2] - radii[1]) / (radii[2]+radii[1])
    rays_ecool = ray_emissivity(rays_T, rays_den, rays_cumulative_taus, radii, dr)
    threshold_temp = np.zeros(len(rays_ecool))
    for i in range(len(rays_ecool)):
        threshold_temp[i] = np.sum(rays_ecool[i])
    return threshold_temp