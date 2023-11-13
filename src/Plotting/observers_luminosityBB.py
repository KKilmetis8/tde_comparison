"""
Created on Mon Oct 9 2023

@author: konstantinos, paola 

Calculate the luminosity that we will use in the blue (BB) curve as sum of multiple BB.

NOTES FOR OTHERS:
- All the functions have to be applied to a CELL
- arguments are in cgs, NOT in log.
- make changes in VARIABLES: frequencies range, fixes (number of snapshots) anf thus days
"""

# Vanilla imports
import numpy as np
import matplotlib.pyplot as plt

# Chocolate imports
from OLD stuff.thermR import get_photosphere
from src.Optical_Depth.opacity_table import opacity
import luminosityBB

# Constants
c = 2.99792458e10 #[cm/s]
h = 6.62607015e-27 #[gcm^2/s]
Kb = 1.380649e-16 #[gcm^2/s^2K]
alpha = 7.5646 * 10**(-15) # radiation density [erg/cm^3K^4]
Rsol_to_cm = 6.957e10

###
# FUNCTIONS
###

def select_observer(obs_array):
    selected_observers = []
    for i in range(0,192, 48):
        selected_observers.append(obs_array[i])
    print(np.shape(select_observer))
    return np.array(selected_observers)

###
# MAIN
###

m = 4
n_min = 1e12 
n_max = 1e18
n_spacing = 10000
x_arr = luminosityBB.log_array(n_min, n_max, n_spacing)
snap_index = 0

snapshots, days = luminosityBB.select_fix(m)
fld_data = np.loadtxt('reddata_m'+ str(m) +'.txt')
fix = snapshots[snap_index]
luminosity_fld_fix = fld_data[1]
n_arr = 10**x_arr

rays_den, rays_T, rays_tau, photosphere, radii = get_photosphere(fix, m)
dr = (radii[1] - radii[0]) * Rsol_to_cm
volume = 4 * np.pi * radii**2 * dr

# rays_den = select_observer(rays_den)
# rays_T = select_observer(rays_T)
# rays_tau = select_observer(rays_tau)
# photosphere = select_observer(photosphere)
# radii = select_observer(radii)
# volume = select_observer(volume)


lum_tilde_n = []
for j in range(0,len(rays_den), 48):
    lum_n_single_obs = np.zeros(len(x_arr))
    for i in range(len(rays_tau[j])):        
        # Temperature, Density and volume: np.array from near to the BH to far away. 
        # Thus we will use negative index in the for loop.
        # tau: np.array from outside to inside.      
        T = rays_T[j][-i]
        rho = rays_den[j][-i] 
        opt_depth = rays_tau[j][i]
        cell_vol = volume[-i]

        # Ensure we can interpolate
        rho_low = np.exp(-22)
        T_low = np.exp(8.77)
        T_high = np.exp(17.8)
        if rho < rho_low or T < T_low or T > T_high:
            continue
        
        for x_index in range(len(x_arr)): #we need linearspace
            freq = 10**x_arr[x_index]
            lum_n_cell = luminosityBB.luminosity_n(T, rho, opt_depth, cell_vol, freq)
            lum_n_single_obs[x_index] += lum_n_cell
        
    # Normalise with the bolometric luminosity from red curve (FLD)
    const_norm = luminosityBB.normalisation(lum_n_single_obs, x_arr, luminosity_fld_fix[snap_index])
    lum_tilde_n_single_obs = lum_n_single_obs * const_norm
    
    lum_tilde_n.append(lum_tilde_n_single_obs)
    print('ray:', j)

lum_tilde_n = np.array(lum_tilde_n)
tot = np.zeros(len(x_arr))
for i in range(len(x_arr)):
    tot[i]= lum_tilde_n[0][i]+lum_tilde_n[1][i] +lum_tilde_n[2][i] +lum_tilde_n[3][i]


print('rays shape: ', np.shape(lum_tilde_n))

fig, ax = plt.subplots()
ax.plot(n_arr, tot)
ax.plot(n_arr, lum_tilde_n[0], label = 'observer ')
ax.plot(n_arr, lum_tilde_n[1], label = 'observer ')
ax.plot(n_arr, lum_tilde_n[2], label = 'observer ')
ax.plot(n_arr, lum_tilde_n[3], label = 'observer ')
plt.xlabel(r'$log_{10}\nu$ [Hz]')
plt.ylabel(r'$log_{10}\tilde{L}_\nu$ [erg/sHz]')
plt.loglog()
plt.grid()
#plt.savefig('Ltildan_m' + str(m) + '_snap' + str(fix))
plt.show()

# plt.figure()
# plt.plot(n_arr, n_arr * lum_tilde_n)
# plt.xlabel(r'$log_{10}\nu$ [Hz]')
# plt.ylabel(r'$log_{10}(\nu\tilde{L}_\nu)$ [erg/s]')
# plt.loglog()
# plt.grid()
# plt.savefig('n_Ltildan_m' + str(m) + '_snap' + str(fix))
# plt.show()