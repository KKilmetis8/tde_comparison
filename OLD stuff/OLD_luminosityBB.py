#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 16:02:14 2023

@author: konstantinos, paola 

Calculate the luminosity NOT normalized that we will use in the blue (BB) curve.

NOTES FOR OTHERS:
- All the functions have to be applied to a CELL
- arguments are in cgs, NOT in log.
- make changes in VARIABLES: frequencies range, fixes (number of snapshots) anf thus days
"""

# Vanilla imports
import numpy as np
import matplotlib.pyplot as plt

# Chocolate imports
from Luminosity.photosphere import get_photosphere
from src.Optical_Depth.opacity_table import opacity

# Constants
c = 2.9979e10 #[cm/s]
h = 6.6261e-27 #[gcm^2/s]
Kb = 1.3806e-16 #[gcm^2/s^2K]
alpha = 7.5646 * 10**(-15) # radiation density [erg/cm^3K^4]
Rsol_to_cm = 6.957e10



###
##
# VARIABLES
##
###

# Frequencies [Hz]
n_min = 1e12 
n_max = 1e19
n_spacing = 100
n_array = np.linspace(n_min, n_max, num = n_spacing)
n_logspace = np.linspace(np.log10(n_min), np.log10(n_max), num = n_spacing)

# Snapshots
fixes4 = [177]
fixes4.append(np.arange(233,264))
days4 = [0.5, 1.015, 1.025, 1.0325, 1.0435, 1.0525, 1.06, 1.07, 1.08, 1.0875, 1.0975, 1.1075, 1.115, 1.125, 1.135, 1.1425, 1.1525, 1.1625, 1.17, 1.18, 1.19, 1.1975, 1.2075, 1.2175, 1.2275, 1.235, 1.245, 1.255, 1.2625, 1.2725, 1.2825, 1.29] #t/t_fb
# days4 = [2, 4.06,4.1,4.13,4.17,4.21,4.24,4.28,4.32,4.35,4.39,4.43,4.46,4.5,4.54,4.57,4.61,4.65,4.68,4.72,4.76,4.79,4.83,4.87,4.91,4.94,4.98,5.02,5.05,5.09,5.13,5.16] #days
fixes6 = [844, 881, 898, 925, 950, 1006]
days6 = [1.00325, 1.13975, 1,2025, 1.302, 1.39425, 1.60075] #t/t_fb
# days6 = [40, 45, 48, 52, 55, 64] #days

###
##
# FUNCTIONS
##
###

def emissivity(Temperature, Density, cell_vol):
    """Arguments in CGS. Gives emissivity in a cell. """
    k_planck = opacity(Temperature, Density, 'planck', ln = False)
    emiss = alpha * c * Temperature**4 * k_planck * cell_vol
    return emiss

def find_peak(Temperature):
    """Find n peak with Wien law."""
    # Wien law
    const_npeak = 5.879e10 # [Hz/K]
    npeak = const_npeak * Temperature
    return npeak # used to be 20 &*

# def planck_fun_n_cell(Temperature: float, n: float) -> float:
#     """ Planck function in a cell. """
#     const = 2*h/c**2
#     peak = find_peak(Temperature)
#     # if n> 20*peak:
#     #     fun = 0
#     # else:
#     fun = const * n**3 / (np.exp(h*n/(Kb*Temperature))-1)
#     return fun

def planck_fun_cell(Temperature: float) -> float:
    """
    Bolometric planck function in a cell. 
    Integrates, so set integrate = True, just saying.
    """
    planck_fun_a_array = []
    # peak = find_peak(Temperature)
    
    for a in n_logspace:
        planck_fun_a_array_single = plank_logspace(Temperature, a, 
                                                   integrate = True)
        planck_fun_a_array.append(planck_fun_a_array_single)
        
    # Integrate
    planck_fun_a_array = np.array(planck_fun_a_array) # Make arr, to intergrate
    fun = np.trapz(planck_fun_a_array, n_logspace)
    return fun

def luminosity_a(Temperature: float, Density: float, tau: float, volume: float,
                 a:int):
    """
    Temperature, Density and volume: np.array from near to the BH to far away. 
    Thus we will use negative index in the for loop.
    tau: np.array from outside to inside.
    n is the frequency.

    We obtain luminosity (at a chosen frequency a (log(n)) ) in a cell.
    """
    epsilon = emissivity(Temperature, Density, volume)
    lum_cell = epsilon * plank_logspace(Temperature, a, integrate = False) \
                * np.exp(-tau)
    return (lum_cell/planck_fun_cell(Temperature))

# def luminosity(Temperature: float, Density: float, tau: float, volume: float) -> int:
#     """Gives NOT normalised bolometric luminosity in a cell."""
#     lum_n_array = []
#     peak = find_peak(Temperature)
#     n_arr = np.linspace(n_min,peak,n_spacing)
#     for n in n_arr:
#         value = luminosity_n(Temperature, Density, tau, volume, n)
#         lum_n_array.append(value)
#     lum_n_array = np.array(lum_n_array)
#     lum = np.trapz(lum_n_array, n_arr)
#     return lum
    

def final_normalisation(L_array: np.array, luminosity_fld: float) -> float:
    """ Find the normalisation constant from FLD model for L_tilde_nu (which is a function of lenght = len(n_array)). """  
    L = np.trapz(L_array, n_logspace)
    norm = luminosity_fld / L
    print('const normalisation: ', norm)
    return  norm

def plank_logspace(T, a, integrate):
    const = 2*h/c**2
    P = 10**(3*a) / (np.exp(h * 10**a / (Kb * T)) -1)
    P *= const
    
    # Change of variables for integration
    if integrate:
        P *= 10**a * np.log(10)

    return P


######
# MAIN
#####
if __name__ == "__main__":
    m = 4
    fld_data = np.loadtxt('reddata_m'+ str(m) +'.txt')
    luminosity_fld_fix = fld_data[1]

    fix_index = len(days4)-1
    rays_den, rays_T, rays_tau, photosphere, radii = get_photosphere(fixes4[fix_index], m)
    dr = (radii[1] - radii[0]) * Rsol_to_cm
    volume = 4 * np.pi * radii**2 * dr  / 192

    lum_a = np.zeros(len(n_logspace))
    for j in range(len(rays_den)):
        for i in range(len(rays_tau[j])):              
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

            for n_index in range(len(n_array)):
                lum_a_cell = luminosity_a(T, rho, opt_depth, cell_vol, n_logspace[n_index])
                lum_a[n_index] += lum_a_cell
        
        print('ray:', j)

    # ANOTHER TRY TO NORMALISE
    const_norm = final_normalisation(lum_a, luminosity_fld_fix[fix_index])
    lum_tilde_a = lum_a *  const_norm

    check = np.trapz(lum_tilde_a, n_logspace)
    check="{:.2e}".format(check) #scientific notation
    print('bolometric L', check)

    with open('Ltilda_m'+ str(m) + '.txt', 'a') as f:
        # f.write(' '.join(map(str, n_logspace)) + '\n')
        f.write('#snap '+ str(fixes4[fix_index])+'\n')
        f.write(' '.join(map(str, lum_tilde_a)) + '\n')
        f.close()

    # from src.Utilities.finished import finished
    # finished()
    #%% Plotting
    fig, ax = plt.subplots()
    ax.plot(n_logspace, lum_tilde_a)
    plt.xlabel(r'$log\nu$ [Hz]')
    plt.ylim(1e38, 1e43)
    plt.yscale('log')
    plt.ylabel(r'$log\tilde{L}_\nu$ [erg/s]')
    plt.grid()
    plt.text(14, 1e39, r'$t/t_{fb}:$ ' + f'{days4[fix_index]}\n B: {check}')
    plt.savefig('Ltilda_m' + str(m) + '_snap' + str(fixes4[fix_index]))
    plt.show()
    ax.axvline(15, color = 'tab:orange')
    ax.axvline(17, color = 'tab:orange')
    ax.axvspan(15, 17, alpha=0.5, color = 'tab:orange')




