#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 20:26:06 2023

@author: konstantinos

Calculates the optical depth. 
Assuming radiation escapes in the z direction.
Uses that to get the cooling time. 

ALL Z NEED TO BE BEFORE ALL R FOR MY INTEGRATOR TO WORK
"""

# Vanilla Imports
import numpy as np
import numba
# Custom Imports
from src.Calculators.casters import THE_CASTER, THE_SMALL_CASTER
from src.Calculators.romberg import romberg
from src.Extractors.time_extractor import days_since_distruption
from src.Optical_Depth.opacity_table import opacity
from scipy.interpolate import RegularGridInterpolator
    
alice = False

if not alice:
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import colorcet
    import matplotlib.patheffects as PathEffects
    AEK = '#F1C410' # Important color
    
# Constants
G = 6.6743e-11 # SI
Msol = 1.98847e30 # kg
Rsol = 6.957e8 # m
Msol_to_g = 1.989e33
Rsol_to_cm = 6.957e10
t = np.sqrt(Rsol**3 / (Msol*G )) # Follows from G=1
# Need for PW pot
c_sim = 3e8 * t/Rsol # c in simulator units.

@numba.njit
def opacity_estimate(rho,T):
    # In CGS
    X = 0.7 # x is actually 0.7
    constant = 3.6e22
    kt = 0.2 * (1+X) # Thompson 
    k = constant * (1+X) * rho * T**(-7/2) + kt # Free free thompson
    return k

def opt_depth(z, r, rho, T):
    rz = rho((z,r))
    Tz = T((z,r))
    tau = rz * opacity_estimate(rz, Tz)
    return tau

def scale_height(zs, rs, d, zmax = False):
    h = np.zeros_like(rs)
    idxs = np.zeros_like(rs)
    
    for i, r in enumerate(rs):
        # Get the densities for said r
        rhos = d[:,i]
        
        if zmax:
            # Find in which z, this is maximum
            idx = np.argmax(rhos)
            h[i] = zs[idx]
            idxs[i] = idx
        else:
            first = rhos[0]
            for j, rho in enumerate(rhos):
                if rho > first / np.exp(1):
                    idx = j
                    h[i] = zs[idx]
                    idxs[i] = idx
                    break
                idx = j
                h[i] = zs[idx]
                idxs[i] = idx
                
    return h, idxs
        
def z_rho_T(m, fix, pixel_num, max_z, alice):
    # BH specific constants
    Mbh = 10**m
    Rt =  Mbh**(1/3) # Msol = 1, Rsol = 1
    apocenter = 2 * Rt * Mbh**(1/3)
    t_fall = 40 * (Mbh/1e6)**(0.5) # days EMR+20 p13
    # rg = 2*Mbh/c**2
    
    # Choose snapshot
    fix = str(fix)

    if alice:
        if m==4:
            folder = 'tde_data2new/snap_' + fix
        if m==6:
            folder = 'tde_data/snap_' + fix
    else:
        folder = fix
        
    days = np.round(days_since_distruption(folder + '/snap_'+fix+'.h5')/t_fall,2) 
    if alice and m==6:
        M = np.load(folder + '/Mass__' + fix + '.npy')
    else:
        M = np.load(folder + '/Mass_' + fix + '.npy')
    
    # # CM Position Data
    X = np.load(folder + '/CMx_' + fix + '.npy')
    Y = np.load(folder + '/CMy_' + fix + '.npy')
    R = np.sqrt(X**2 + Y**2)
    Z = np.load(folder + '/CMz_' + fix + '.npy')

    # Import Density
    Den = np.load(folder + '/Den_' + fix + '.npy')
    T = np.load(folder + '/T_' + fix + '.npy')

    # Convert to cgs
    converter = Msol_to_g / Rsol_to_cm**3
    Den *=  converter
    # Z *= Rsol_to_cm
    
    # EVOKE
    # start from rg or something in z
    zs = np.linspace(0.2 * Rt, max_z, num = pixel_num)
    radii = np.linspace(0.1*2*Rt, apocenter, num = pixel_num)
    Den_cast = THE_CASTER(zs, Z, radii, R, 
                          Den, weights = M, avg = False)
    T_cast = THE_CASTER( zs, Z, radii, R,
                        T, weights = M, avg = True)
    
    Den_cast = np.nan_to_num(Den_cast)
    T_cast = np.nan_to_num(T_cast)
    
    # Ratio
    ie = np.load(folder + '/IE_' + fix + '.npy')
    rad = np.load(folder + '/Rad_' + fix + '.npy')
    ratio = ie/rad
    ratio_cast = THE_SMALL_CASTER(radii, R, 
                          ratio, weights = M, avg = True)
    ratio_cast = np.nan_to_num(ratio_cast)
    
    return zs, radii, Den_cast, T_cast, days, ratio_cast

#%% Get Z, rho, T
fixes4 = np.arange(177, 263+1)
fixes6 = np.arange(683, 1008+1)
max_z4 = 50
max_z6 = 300
pixel_num4 = 100
pixel_num6 = 100
if alice:
    tc4 = []
    tc6 = []
    for fix4 in fixes4:
        # Cast
        z4, r4, d4, t4, days4, ratio4 = z_rho_T(4, fix4, pixel_num4, max_z4, alice)
        
        # Interpolate
        d4_inter = RegularGridInterpolator((z4, r4), d4)
        t4_inter = RegularGridInterpolator((z4, r4), t4)
        
        # Calculate Optical Depth
        opt4 = np.zeros_like(r4)
        for i, r in enumerate(r4):
            Mbh4 = 10**4
            Rt4 =  Mbh4**(1/3) # Msol = 1, Rsol = 1
            opt4[i] = romberg(0.2 * Rt4 ,max_z4, opt_depth, r,
                                  d4_inter, t4_inter) # Rsol / cm 
        
        # Constants
        converter = Rsol_to_cm
        t_fall_4 = 40 * (Mbh4/1e6)**(0.5) # days EMR+20 p13
        apocenter4 = 2 * Rt4 * Mbh4**(1/3)

        # Scale height, convert to cgs
        h4, _ = scale_height(z4, r4, d4) 
        # h4 = max_z4
        h4 *= converter
        sec_to_days = 60*60*24
        
        # Cooling time
        t_cool_4 = h4*opt4/(2*c_sim) # sim units -> days
        t_cool_4 *=  sec_to_days / t_fall_4
        
        # Store
        tc4.append( t_cool_4)
    print('4 is finished, moving on to 6')
    # Again
    for fix6 in fixes6:
        # Cast
        z6, r6, d6, t6, days6, ratio6 = z_rho_T(6, fix6, pixel_num6, max_z6, alice)
        
        # Interpolate
        d6_inter = RegularGridInterpolator((z6, r6), d6)
        t6_inter = RegularGridInterpolator((z6, r6), t6)
        
        # Calculate Optical Depth
        opt6 = np.zeros_like(r6)
        for i, r in enumerate(r6):
            Mbh6 = 10**6
            Rt6 =  Mbh6**(1/3) # Msol = 1, Rsol = 1
            opt6[i] = romberg(0.2 * Rt6 ,max_z6, opt_depth, r,
                                  d6_inter, t6_inter) # Rsol / cm 
        
        # Constants
        converter = Rsol_to_cm
        t_fall_6 = 60 * (Mbh6/1e6)**(0.5) # days EMR+20 p13
        apocenter6 = 2 * Rt6 * Mbh6**(1/3)

        # Scale height, convert to cgs
        h6, _ = scale_height(z6, r6, d6) 
        # h6 = max_z6
        h6 *= converter
        sec_to_days = 60*60*24
        
        # Cooling time
        t_cool_6 = h6*opt6/(2*c_sim) # sim units -> days
        t_cool_6 *=  sec_to_days / t_fall_6
        
        # Store
        tc6.append( t_cool_6)
        
    # Save
    savepath = 'products/cooling-time/'
    np.save(savepath + 'tc4', tc4)
    np.save(savepath + 'tc6', tc6)

    
else:
    fix4 = 210
    fix6 = 820
    z4, r4, d4, t4, days4, ratio4 = z_rho_T(4, fix4, pixel_num4, max_z4, alice)
    z6, r6, d6, t6, days6, ratio6 = z_rho_T(6, fix6, pixel_num6, max_z6, alice)
    
    # Interpolate
    d4_inter = RegularGridInterpolator((z4, r4), d4)
    t4_inter = RegularGridInterpolator((z4, r4), t4)
    d6_inter = RegularGridInterpolator((z6, r6), d6)
    t6_inter = RegularGridInterpolator((z6, r6), t6)

    
    #%% Integrate
    opt4 = np.zeros_like(r4)
    for i, r in enumerate(r4):
        Mbh = 10**4
        Rt =  Mbh**(1/3) # Msol = 1, Rsol = 1
        opt4[i] = romberg(0.2 * Rt ,max_z4, opt_depth, r,
                              d4_inter, t4_inter) # Rsol / cm 
        
    opt6 = np.zeros_like(r6)
    for i, r in enumerate(r6): 
        Mbh = 10**6
        Rt =  Mbh**(1/3) # Msol = 1, Rsol = 1
        opt6[i] = romberg(0.2 * Rt,max_z6, opt_depth, r,
                              d6_inter, t6_inter) # Rsol / cm

#%% Calculate cooling time
# I realize this code is clunkster central but it is a proof of concept

    # Convert zs to cgs
    converter = Rsol_to_cm
    # c = 3e10
    
    # Get t_fall in days
    Mbh = 10**4
    Rt4 =  Mbh**(1/3) # Msol = 1, Rsol = 1
    t_fall_4 = 40 * (Mbh/1e6)**(0.5) # days EMR+20 p13
    apocenter4 = 2 * Rt4 * Mbh**(1/3)
    # sim_to_days = t / (60*60*24)
    
    # Calculate scale height, convert to cgs
    h4, idxs4 = scale_height(z4, r4, d4) 
    # h4 = max_z4
    h4 *= converter
    
    sec_to_days = 60*60*24
    # Cooling time
    t_cool_4 = h4*opt4/(2*c_sim) # sim units -> days
    t_cool_4 *=  sec_to_days / t_fall_4
    
    # Again for 10^6
    Mbh = 10**6
    Rt6 =  Mbh**(1/3) # Msol = 1, Rsol = 1
    t_fall_6 = 40 * (Mbh/1e6)**(0.5) # days EMR+20 p13
    apocenter6 = 2 * Rt6 * Mbh**(1/3)
    
    h6, idxs6 = scale_height(z6, r6, d6)
    #    h6 = max_z6
    h6 *= converter
    
    t_cool_6 = h6*opt6/(2*c_sim) # sim units -> days
    t_cool_6 *= sec_to_days / t_fall_6
    
    # Using the scale height, calculate the ratio for the relevant range
    # ratio4_mean = np.zeros_like(r4)
    # for i, idx in enumerate(idxs4):
    #     idx = int(idx)
    #     ratio4_mean[i] = np.mean(ratio4[:idx,i])
    # ratio6_mean = np.zeros_like(r6)    
    # for i, idx in enumerate(idxs6):
    #     idx = int(idx)
    #     ratio6_mean[i] = np.mean(ratio6[:idx,i])    
    
    # Import internal and radiative energies
    t_cool_4_ratio = h4 * opt4 / (c_sim * (1 + ratio4))
    t_cool_4_ratio *= sec_to_days / t_fall_4
    
    t_cool_6_ratio = h6 * opt6 / (c_sim * (1 + ratio6))
    t_cool_6_ratio *= sec_to_days / t_fall_6
        
    #%% Plotting
if alice == False:
    fig, ax = plt.subplots(2,1, figsize = (8,6), tight_layout = True)
    ax[0].plot(r4/apocenter4, t_cool_4, 
             '-h', color = AEK, label = '$10^4 M_\odot$', 
             markersize = 5)
    ax[0].plot(r6/apocenter6, t_cool_6, 
             '-h', color = 'k', label = '$10^6 M_\odot$',
             markersize = 5)
    ax[0].set_yscale('log')
    ax[0].set_xlabel('r [r/R$_a$]')
    ax[0].set_ylabel(r'$\left[ t_c/t_{FB} \right]$')
    ax[0].legend()
    ax[0].grid()

    
    ax[0].axvline(Rt4/apocenter4, color = 'r', linestyle = 'dashed')
    ax[0].axvline(Rt6/apocenter6, color = 'maroon', linestyle = 'dashed')
    ax[0].text(0.07, 0.7, '$10^4 M_\odot$ $R_P$', 
               color = 'r', rotation = 90,  
               transform = ax[0].transAxes)
    ax[0].text(0.02, 0.7, '$10^6 M_\odot$ $R_P$', 
               color = 'Maroon', rotation = 90, 
               transform = ax[0].transAxes)
    ax[0].set_title('Assuming ugas/urad = 1')
    
    # With the ratio
    ax[1].plot(r4/apocenter4, t_cool_4_ratio, 
             '-h', color = AEK, label = '$10^4 M_\odot$', 
             markersize = 5)
    ax[1].plot(r6/apocenter6, t_cool_6_ratio, 
             '-h', color = 'k', label = '$10^6 M_\odot$',
             markersize = 5)
    ax[1].set_yscale('log')
    ax[1].set_xlabel('r [r/R$_a$]')
    ax[1].set_ylabel(r'$\left[ t_c/t_{FB} \right]$')
    ax[1].legend()
    ax[1].grid()

    
    ax[1].axvline(Rt4/apocenter4, color = 'r', linestyle = 'dashed')
    ax[1].axvline(Rt6/apocenter6, color = 'maroon', linestyle = 'dashed')
    ax[1].text(0.07, 0.7, '$10^4 M_\odot$ $R_P$', 
               color = 'r', rotation = 90,  
               transform = ax[1].transAxes)
    ax[1].text(0.02, 0.7, '$10^6 M_\odot$ $R_P$', 
               color = 'Maroon', rotation = 90, 
               transform = ax[1].transAxes)
    ax[1].set_title('Assuming ugas/urad is not 1')
    
        
    from src.Utilities.finished import finished
    finished()

#%% Interpolation
# if alice == False:
    plt.rcParams['text.usetex'] = True
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.facecolor']= 	'whitesmoke'
    
    fig, ax = plt.subplots(2,1, figsize = (5,6), tight_layout = True)

    ax[0].scatter(z6, d6[:,5], c='k', marker='h', label='True Points')
    ax[0].plot(z6, d6_inter((z6, r6[5])), c=AEK, label = 'Interpolation')

    ax[1].scatter(z4, d4[:,5], c='k', marker='h', label='True Points')
    ax[1].plot(z4, d4_inter((z4, r4[5])), c=AEK, label = 'Interpolation')

    ax[0].set_title(r'$10^6 M_\odot$' )
    ax[0].set_xlabel(r'$z$ [$R_\odot $]')
    ax[0].set_ylabel(r'$\rho$ [$M_\odot/R_\odot^3$]')
    ax[0].grid()
    ax[0].legend()
    ax[1].set_xlabel(r'$z$ [$R_\odot $]')
    ax[1].set_ylabel(r'$\rho$ [$M_\odot/R_\odot^3$]')
    ax[1].set_title(r'$10^4 M_\odot$' )
    ax[1].grid()
    ax[1].legend()
    
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    
    fig, ax = plt.subplots(2,1, figsize = (5,6), tight_layout = True, sharex = True)

    ax[0].set_ylabel(r'$\rho$ [$M_\odot/R_\odot^3$]')
    ax[0].set_title(r'$10^6 M_\odot$' )
    ax[0].grid()
    ax[1].set_ylabel(r'$\rho$ [$M_\odot/R_\odot^3$]')
    ax[1].set_title(r'$10^4 M_\odot$' )
    ax[1].grid()
    plt.xlabel(r'$r$ [$R_\odot $]')
    
    ax[0].scatter(r6/apocenter6, d6[5,:], c='k', marker='h', label='True Points')
    ax[0].plot(r6/apocenter6, d6_inter((z6[5], r6)), c=AEK, label = 'Interpolation')

    ax[1].scatter(r4/apocenter4, d4[5,:], c='k', marker='h', label='True Points')
    ax[1].plot(r4/apocenter4, d4_inter((z4[5], r4)), c=AEK, label = 'Interpolation')
        
#%% Explore optical depth
    opt6 = np.nan_to_num(opt6)
    opt6 *= Rsol_to_cm
    opt4 = np.nan_to_num(opt4)
    opt4 *= Rsol_to_cm
    
    #%%
    fig, ax = plt.subplots(1,2, figsize = (10,5), tight_layout = True, sharex = True)
    ax[0].plot(r6/apocenter6, opt6, c = 'k', label = '$10^6 M_\odot$')
    ax[0].plot(r4/apocenter4, opt4, c = AEK,  label = '$10^4 M_\odot$')
    
    ax[0].set_title('Optical Depth', fontsize = 25 )
    ax[0].set_yscale('log')
    ax[0].set_ylabel(r'$\tau (r) $', fontsize = 18) # '/ \tau_{MAX}$')
    ax[0].set_xlabel(r'Distance from BH $[r/R_a]$', fontsize = 18)
    ax[0].grid()
    ax[0].legend(loc = 'lower right')
    
    
    # uint/urad
    ax[1].plot(r6/apocenter6, ratio6, c = 'k',  label = '$10^6 M_\odot$')
    ax[1].plot(r4/apocenter4, ratio4, c = AEK,  label = '$10^4 M_\odot$')
    
    ax[1].set_title('Specific Energies Ratio', fontsize = 25)
    ax[1].set_yscale('log')
    ax[1].set_ylabel(r'$u_{Gas} / u_{Rad}$', fontsize = 18)
    ax[1].set_xlabel(r'Distance from BH $[r/R_a]$' , fontsize = 18)
    ax[1].grid()
    ax[1].legend(loc = 'lower right')
    
    
