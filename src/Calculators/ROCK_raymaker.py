#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 16:22:40 2023

@author: konstantinos
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from astropy.coordinates import cartesian_to_spherical
from src.Opacity.opacity_table import opacity
import numba

# Constants
NSIDE = 4
G = 6.6743e-11 # SI
Msol = 1.98847e30 # kg
Rsol = 6.957e8 # m
t = np.sqrt(Rsol**3 / (Msol*G )) # Follows from G = 1
c = 3e8 * t/Rsol # simulator units. Need these for the PW potential
c_cgs = 3e10 # [cm/s]
Msol_to_g = 1.989e33 # [g]
Rsol_to_cm = 6.957e10 # [cm]
den_converter = Msol_to_g / Rsol_to_cm**3
en_den_converter = Msol_to_g / (Rsol_to_cm  * t**2 ) # Energy Density converter

fix = '844'
m = 6
prune = 1
X = np.load( str(m) + '/'  + fix + '/CMx_' + fix + '.npy')[::prune]
Y = np.load( str(m) + '/'  + fix + '/CMy_' + fix + '.npy')[::prune]
Z = np.load( str(m) + '/'  + fix + '/CMz_' + fix + '.npy')[::prune]
Mass = np.load( str(m) + '/'  + fix + '/Mass_' + fix + '.npy')[::prune]
T = np.load( str(m) + '/'  + fix + '/T_' + fix + '.npy')[::prune]
Den = np.load( str(m) + '/'  + fix + '/Den_' + fix + '.npy')[::prune]
Rad = np.load( str(m) + '/'  +fix + '/Rad_' + fix + '.npy')[::prune]

Rad *= Den 
Rad *= en_den_converter
Den *= den_converter 
# Convert to spherical
R, THETA, PHI = cartesian_to_spherical(X,Y,Z)
# R = R.value 
THETA = THETA.value
PHI = PHI.value

#%% Find observers with Healpix
thetas = np.zeros(192)
phis = np.zeros(192)
observers = []
delta_thetas = []
delta_phis = []
fig, ax = plt.subplots(1,1, subplot_kw=dict(projection="mollweide"))
for i in range(192):
    # Get Observers
    thetas[i], phis[i] = hp.pix2ang(NSIDE, i)
    thetas[i] -= np.pi/2 # Enforce theta in -pi to pi    
    observers.append( (thetas[i], phis[i]) )
    
    # Get their opening angles
    theta_diff = 0
    phi_diff = 0
    corner_angles = []
    # Corner XYZ coordinates
    corner_x, corner_y, corner_z = hp.boundaries(4,i)
    corners = np.zeros((4,3))
    corner_angles = np.zeros((4,2))
    
    # Corner 
    for i in range(4):
        corners[i] = corner_x[i], corner_y[i], corner_z[i]
        corner_angles[i][0] = hp.vec2ang(corners[i])[0][0]
        corner_angles[i][1] = hp.vec2ang(corners[i])[1][0]
        
    # Biggest possible Delta phi and Delta theta
    print(corner_angles.T[1].max() - corner_angles.T[1].min())
    corner_angles_phi_plot = list(corner_angles.T[1] - np.pi)
    corner_angles_theta_plot = list(corner_angles.T[0] - np.pi/2)
    
    # last point to close the rectangle
    corner_angles_phi_plot.append(corner_angles_phi_plot[0]) 
    corner_angles_theta_plot.append(corner_angles_theta_plot[0])
    
    plt.plot(corner_angles_phi_plot, corner_angles_theta_plot, '-x', 
              markersize = 1, c='k', linewidth = 1, zorder = 4,)

    for i in range(0,4):
        theta_diff_temp = np.max(np.abs(corner_angles[i][0] - corner_angles[:][0]))
        phi_diff_temp = np.max(np.abs(corner_angles[i][1] - corner_angles[:][1])) 
        
        if theta_diff_temp > theta_diff:
            theta_diff = theta_diff_temp
            
        if phi_diff_temp > phi_diff:
            phi_diff = phi_diff_temp
            
    delta_thetas.append(theta_diff/2)
    delta_phis.append(phi_diff/2)
plt.plot(PHI[::100] - np.pi, THETA[::100], 'x',  c = 'r', markersize = 0.1, zorder = 2) 


#%% Make rays

rays_T = []
rays_Den = []
rays_R = []
rays_Rad = []

for observer in observers:
    # These are kind of wrong, we should be more explicit about this
    delta_phi = 0.392 / 2 # rad
    delta_theta = 0.339 / 2 # rad
    # numpy thinks in C
    theta_mask = ( observer[0] - delta_theta < THETA) & \
                 ( THETA < observer[0] + delta_theta)
                 
    phi_mask = ( observer[1] - delta_phi < PHI) & \
               ( PHI < observer[1] + delta_phi)
    fluff_mask = ( Den > 1e-17 )
    
    # Mask & Mass Weigh
    # ray_Mass = Mass[theta_mask & phi_mask]
    # sum_mass = np.sum(ray_Mass)
    ray_R = R[theta_mask & phi_mask & fluff_mask] 
    ray_Den = Den[theta_mask & phi_mask  & fluff_mask ]# * ray_Mass / sum_mass
    ray_T = T[theta_mask & phi_mask  & fluff_mask]# * ray_Mass / sum_mass
    ray_Rad = Rad[theta_mask & phi_mask  & fluff_mask]# * ray_Mass / sum_mass
    
    # Sort by r
    bookeeper = np.argsort(ray_R)
    ray_R = ray_R[bookeeper]
    ray_Den = ray_Den[bookeeper]
    ray_T = ray_T[bookeeper]
    ray_Rad = ray_Rad[bookeeper]

    # Keep
    rays_R.append(ray_R)
    rays_Den.append( ray_Den )
    rays_T.append( ray_T)
    rays_Rad.append(ray_Rad)
#%%
def get_kappa(T: float, rho: float, dr: float):
    '''
    Calculates the integrand of eq.(8) Steinberg&Stone22.
    '''    
    # If there is nothing, the ray continues unimpeded
    if rho < np.exp(-49.3):
        return 0
    
    # Stream material, is opaque
    if T < np.exp(8.666): # 5802 K
        if rho > 1e-9:
            return 1 # testing 1
        else:
            return 0
    
    # Too hot: Kramers' law for absorption (planck)
    if T > np.exp(17.876): # 58_002_693
        X = 0.7389
        Z = 0.02
        kplanck =  1.2e26 * Z * (1 + X) * T**(-3.5) * rho #Kramers' bound-free opacity [cm^2/g]
        kplanck *= rho

        Tscatter = np.exp(17.87)
        kscattering = opacity(Tscatter, rho, 'scattering', ln = False)

        oppi = kplanck + kscattering
        tau_high = oppi * dr
        return tau_high 
    
    # Lookup table
    k = opacity(T, rho,'red', ln = False)
    kappar =  k * dr
    
    return kappar

def calc_photosphere(T, rho, rs):
    '''
    Input: 1D arrays.
    Finds and saves the photosphere (in CGS).
    The kappas' arrays go from far to near the BH.
    '''
    threshold = 2/3
    kappa = 0
    kappas = []
    cumulative_kappas = []
    i = -1 # Initialize reverse loop
    while kappa <= threshold and i > -len(T):
        dr = rs[i]-rs[i-1] # Cell seperation
        new_kappa = get_kappa(T[i], rho[i], dr)
        kappa += new_kappa
        kappas.append(new_kappa)
        cumulative_kappas.append(kappa)
        i -= 1
    photo =  rs[i] #i it's negative
    return kappas, cumulative_kappas, photo

def get_photosphere(rays_T, rays_den, rays_R):
    '''
    Finds and saves the photosphere (in CGS) for every ray.

    Parameters
    ----------
    rays_T, rays_den: n-D arrays.
    radii: 1D array.

    Returns
    -------
    rays_kappas, rays_cumulative_kappas: nD arrays.
    photos: 1D array.
    '''
    # Get the thermalisation radius
    rays_cumulative_kappas = []
    rays_photo = np.zeros(len(rays_T))
    
    for i in range(len(rays_T)):
        print('Ray maker ray: ', i) 
        # Isolate each ray
        T_of_single_ray = rays_T[i]
        Den_of_single_ray = rays_den[i]
        R_of_single_ray = rays_R[i]
        # Get photosphere
        _, cumulative_kappas, photo  = calc_photosphere(T_of_single_ray, 
                                                             Den_of_single_ray, 
                                                             R_of_single_ray)

        # Store
        # rays_kappas.append(kappas)
        rays_cumulative_kappas.append(cumulative_kappas)
        rays_photo[i] = photo

    return rays_cumulative_kappas, rays_photo

stop = 192
cumul_kappa, photo =  get_photosphere(rays_T[:stop], rays_Den[:stop], rays_R[:stop])
#%%

@numba.njit
def grad_calculator(ray: np.array, radii: np.array, r_photo): 
    # For a single ray (in logspace) get 
    # the index of radius closest to sphere_radius and the gradE there.
    # Everything is in CGS.
    for i, radius in enumerate(radii):
        if radius > r_photo:
            idx = i - 1 
            break
    
    # For rad
    if idx<0:
        print('Bad Observer, photosphere is the closest point')
        idx=0
        
    step = radii[idx+1] - radii[idx]
    grad_E = (ray[idx+1] - ray[idx]) / step 

    return grad_E, idx

    
def flux_calculator(grad_E, idx, 
                    single_Rad, single_T, single_Den):
    """
    Get the flux for every observer.
    Eevrything is in CGS.

    Parameters: 
    grad_E idx_tot are 1D-array of lenght = len(rays)
    rays, rays_T, rays_den are len(rays) x N_cells arrays
    """
    # We compute stuff OUTSIDE the photosphere
    # (which is at index idx_tot[i])
    idx = idx+1 #  
    Energy = single_Rad[idx]
    max_travel = np.sign(-grad_E) * c_cgs * Energy # or should we keep the abs???
    
    Temperature = single_T[idx]
    Density = single_Den[idx]
    
    # Ensure we can interpolate
    rho_low = np.exp(-45)
    T_low = np.exp(8.77)
    T_high = np.exp(17.8)
    
    # If here is nothing, light continues
    if Density < rho_low:
        return max_travel
    
    # If stream, no light 
    if Temperature < T_low: 
        return 0
    
    # T too high => Kramers'law
    if Temperature > T_high:
        X = 0.7389
        k_ross = 3.68 * 1e22 * (1 + X) * Temperature**(-3.5) * Density #Kramers' opacity [cm^2/g]
        k_ross *= Density
    else:    
        # Get Opacity, NOTE: Breaks Numba
        k_ross = opacity(Temperature, Density, 'rosseland', ln = False)
    
    # Calc R, eq. 28
    R = np.abs(grad_E) /  (k_ross * Energy)
    invR = 1 / R
    # Calc lambda, eq. 27
    coth = 1 / np.tanh(R)
    lamda = invR * (coth - invR)
    # Calc Flux, eq. 26
    Flux = - c_cgs * grad_E  * lamda / k_ross
    
    # Take the minimum between F, cE
    if Flux > max_travel:
        return max_travel
    else:
        return Flux

fluxes = []
for i in range(len(rays_T)):
    print('Flux ray: ', i) 

    # Isolate ray
    single_T = rays_T[i]
    single_Den = rays_Den[i]
    single_Rad = rays_Rad[i]
    single_radii = rays_R[i]
    
    #  Calculate gradE
    grad_E, idx = grad_calculator(single_Rad, single_radii, photo[i])

    # Calculate Flux 
    flux = flux_calculator(grad_E, idx, 
                            single_Rad, single_T, single_Den)
    fluxes.append(flux) # Keep it
    
#%% Convert to luminosity 
lum = np.zeros(len(fluxes))
for i in range(len(fluxes)):
    # Ignore negative flux
    if fluxes[i] < 0:
        continue
    # Convert photo to cm
    r = photo[i] * Rsol_to_cm
    lum[i] = fluxes[i] * 4 * np.pi * r**2

# Average in observers
bol_lum = np.sum(lum)/192

from src.Utilities.finished import finished
finished()
#%% Plot & Print
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.figsize'] = [5 , 4]
plt.rcParams['axes.facecolor']= 	'whitesmoke'
plt.rcParams['figure.figsize'] = [5 , 4]
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
AEK = '#F1C410' # Important color 

# Print bol lum
print('---')
print('L FLD: %.2e' % bol_lum)
print('Should be: 3e42')

# Plot
fig, ax = plt.subplots(1,2, figsize = (10, 6))
ax[0].plot(lum, 'o', c ='k')

# Make pretty
ax[0].set_yscale('log')
ax[0].set_title('FLD Luminosity')
ax[0].set_xlabel('Observers')
ax[0].set_ylabel('Luminosity [erg/s]')

# Plot Photo
ax[1].plot(photo, 'o', c = 'k')

## Make pretty
# My lines
from scipy.stats import gmean
ax[1].axhline(np.mean(photo), linestyle = '--' ,c = 'seagreen')
ax[1].axhline(gmean(photo), linestyle = '--', c = 'darkorange')
#ax[1].text()
# Elad's lines
ax[1].axhline(40 , c = AEK)
ax[1].axhline(700, c= 'lightseagreen')

ax[1].set_yscale('log')
ax[1].set_title('Photosphere')
ax[1].set_ylabel(r'Photosphere Radius [$R_\odot$]')
ax[1].set_xlabel('Observers')

