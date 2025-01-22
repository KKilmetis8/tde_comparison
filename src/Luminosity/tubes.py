#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 19:00:20 2024

@author: konstantinos
|
The new tubes
"""
import warnings
warnings.filterwarnings('ignore')

# Vanilla Imports
import numba
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from astropy.coordinates import cartesian_to_spherical
import matlab.engine
eng = matlab.engine.start_matlab()
from tqdm import tqdm
from scipy.ndimage import uniform_filter1d # does moving mean without fucking the shape up
import scipy.integrate as sci
from scipy.interpolate import griddata


# Custom Imports
import src.Utilities.prelude as c
from src.Opacity.LTE_loader import T_opac_ex, Rho_opac_ex, rossland_ex, plank_ex
from src.Calculators.borderlands import borderlands

@numba.njit
def gradient(arr, x=1.0):
    n = len(arr)
    grad = np.zeros_like(arr, dtype=np.float64)
    # Forward difference at the first point
    grad[0] = (arr[1] - arr[0]) / (x[1] - x[0])
    # Central difference for the interior points
    for i in range(1, n - 1):
        grad[i] = (arr[i + 1] - arr[i - 1]) / 2 * (x[i+1] - x[i-1])
    # Backward difference at the last point
    grad[-1] = (arr[-1] - arr[-2]) / (x[-1] - x[-2])
    return grad

@numba.njit
def spherical_gradients(R, THETA, PHI, Rad, cell_radius):
    gradE_r = gradient(Rad, R) #/ (2*cell_radius)
    gradE_theta = gradient(Rad, THETA) #/ (2*cell_radius)
    gradE_phi = gradient(Rad, PHI) # / (2*cell_radius)
    gradE = gradE_r + 1/R * gradE_theta + 1/(R * np.sin(THETA)) * gradE_phi
    return gradE_r, gradE

@numba.njit
def cartesian_gradients(X, Y, Z, Rad, cell_radius, rhat, neigh_num = 20):
    # Grads Cartesian (how Elad does it)
    Xgrad = gradient(Rad, X) / (2*cell_radius)
    Ygrad = gradient(Rad, Y) / (2*cell_radius)
    Zgrad = gradient(Rad, Z) / (2*cell_radius)
    
    gradE = np.sqrt(Xgrad**2 + Ygrad**2 + Zgrad**2)
    gradE_r = rhat[0] * Xgrad + rhat[1]*Ygrad + rhat[2]*Zgrad
    return gradE_r, gradE

#%% Load & Clean
# Specify snap
m = 5
pre = f'{m}/'
snap = 349

# Data load
X = np.load(f'{pre}{snap}/CMx_{snap}.npy')
Y = np.load(f'{pre}{snap}/CMy_{snap}.npy')
Z = np.load(f'{pre}{snap}/CMz_{snap}.npy')
T = np.load(f'{pre}{snap}/T_{snap}.npy')
Den = np.load(f'{pre}{snap}/Den_{snap}.npy')
Rad = np.load(f'{pre}{snap}/Rad_{snap}.npy')
Vol = np.load(f'{pre}{snap}/Vol_{snap}.npy')

Rad *= Den # Energy/mass -> Energy/Volume

# To spherical
R, THETA, PHI = cartesian_to_spherical(X,Y,Z)
R = R.value
THETA = THETA.value
PHI = PHI.value

# Downsample by costheta parameter
npix = hp.nside2npix(c.NSIDE)
theta, phi = hp.pix2ang(c.NSIDE, range(npix))
outx = np.sin(theta) * np.cos(phi)
outy = np.sin(theta) * np.sin(phi)
outz = np.cos(theta)
outX = np.array([outx, outy, outz]).T
cross_dot = np.matmul(outX,  outX.T )
cross_dot[cross_dot<0] = 0
cross_dot *= 4/192

# Freq range
f_min = c.kb * 1e3 / c.h
f_max = c.kb * 3e13 / c.h
f_num = 1_000
freqs = np.logspace(np.log10(f_min), np.log10(f_max), f_num)
#%% Find observers
thetas = np.zeros(npix)
phis = np.zeros(npix)
observers = []
corners = []
for i in range(npix):
    thetas[i], phis[i] = hp.pix2ang(c.NSIDE, i)
    thetas[i] -= np.pi/2 # Enforce theta in -pi to pi    
    observers.append( (thetas[i], phis[i]))
    
    # Get their opening angles
    theta_diff = 0
    phi_diff = 0
    corner_angles = []
    
    # Corner XYZ coordinates
    corner_x, corner_y, corner_z = hp.boundaries(c.NSIDE,i)
    corners_xyz = np.zeros((4,3))
    corner_angles = np.zeros((4,2))
    
    # Corner 
    for j in range(4):
        corners_xyz[j] = corner_x[j], corner_y[j], corner_z[j]
        corner_angles[j][0] = hp.vec2ang(corners_xyz[j])[0][0] - np.pi/2
        corner_angles[j][1] = hp.vec2ang(corners_xyz[j])[1][0] 
        
    # Fix horizontals
    if any(corner_angles.T[1] > 6):
        for k, corner in enumerate(corner_angles.T[1]):
            if corner > 6.28 or corner < 0.4: # 2 * np.pi for healpix
                corner_angles[k][1] = 6.27
    corners.append(corner_angles.T)

#%% Make rays
class ray:
    def __init__(self, T, Den, R, THETA, PHI, Rad, X, Y, Z, Vol, theta, phi):
        # Convert to cgs
        self.T = T
        self.Den = Den * c.den_converter
        self.R = R * c.Rsol_to_cm
        self.THETA = THETA
        self.PHI = PHI 
        
        self.Rad = Rad * c.en_den_converter
        self.X = X * c.Rsol_to_cm
        self.Y = Y * c.Rsol_to_cm
        self.Z = Z * c.Rsol_to_cm
        self.Vol = Vol / c.Rsol_to_cm**3
    
        # Needed for r vector, pointing from BH
        self.rhat_x = np.sin(theta) * np.cos(phi)
        self.rhat_y = np.sin(theta) * np.sin(phi)
        self.rhat_z = np.cos(theta)
        self.rhat = np.array([self.rhat_x, self.rhat_y, self.rhat_z])
        
    def sort(self,):
        sorter = np.argsort(self.R)
        self.T = self.T[sorter]
        self.Den = self.Den[sorter]
        self.R = self.R[sorter]
        self.Rad = self.Rad[sorter]
        self.X = self.X[sorter]
        self.Y = self.Y[sorter]
        self.Z = self.Z[sorter]
        self.Vol = self.Vol[sorter]

    def calc_taus(self, eng):
        # Interpolate on table for opacities
        sigma_rossland = np.exp(eng.interp2(T_opac_ex, Rho_opac_ex, rossland_ex.T,
                                     np.log(self.T), np.log(self.Den), 
                                     'linear', 0))
        self.sigma_rossland = np.array(sigma_rossland)[0]
        
        sigma_plank = np.exp(eng.interp2(T_opac_ex, Rho_opac_ex, plank_ex.T,
                                     np.log(self.T), np.log(self.Den), 
                                     'linear', 0))
        self.sigma_plank = np.array(sigma_plank)[0]
        
        # Cumulative sum for optical depth
        R_flipped = np.flipud(self.R.T)
        sigma_ross_flipped = np.flipud(self.sigma_rossland) 
        self.tau_red = - np.flipud(sci.cumulative_trapezoid(sigma_ross_flipped, 
                                                       R_flipped, 
                                                       initial = 0)).T
        
        sigma_eff_flipped = np.sqrt(3 * np.flipud(self.sigma_plank) * np.flipud(self.sigma_rossland)) 
        self.tau_eff = - np.flipud(sci.cumulative_trapezoid(sigma_eff_flipped, 
                                                       R_flipped, 
                                                       initial = 0)).T
        self.tau_eff[self.tau_eff > 30] = 30 # Overflow protection

    
    def calc_colorsphere(self, photostop = 2/3, colorstop = 5):
        ''' Keeps indices '''
        self.colorsphere = np.argmin(np.abs(self.tau_eff - colorstop))
            
    def calc_flux(self):
        cell_radius = 0.5 * self.Vol**(1/3)
        
        # Grads
        gradE_r, gradE = cartesian_gradients(self.X, self.Y, self.Z, 
                                              self.Rad, cell_radius, self.rhat)

        # Krumholz et al. '07 
        R_lamda = gradE /  (self.sigma_rossland * self.Rad) # eq. 28
        R_lamda[R_lamda < 1e-10] = 1e-10 # R is unitless here
        invR = 1 / R_lamda
        coth = 1 / np.tanh(R_lamda) 
        fld_factor = 3 * invR * (coth - invR) # eq. 27

        flux = - c.c * gradE_r  * fld_factor / self.sigma_rossland # eq. 26
        smoothed_flux = uniform_filter1d(flux, 7)
        # Get photosphere
        try:
            self.photosphere  = np.where( ((smoothed_flux>0) & (self.tau_red<2/3) ))[0][0]
            # self.photosphere = np.min(photo_cand)
        except:
            print('\n eladb')
            self.photosphere = 0
        L_photo = 4 * np.pi * self.R[self.photosphere]**2 * smoothed_flux[self.photosphere]
        
        if L_photo < 0: # Ensure max travel is picked if L<0
            L_photo = 1e100
        max_travel = 4 * np.pi * c.c * self.Rad[self.photosphere] * self.R[self.photosphere]**2
        self.L_photo = np.min([max_travel, L_photo])
    
    def calc_spectra(self, freqs, cross_dot):
        spectrum = 0
        for i in range(self.colorsphere, len(self.R)):
            wien = np.exp(c.h * freqs / (c.kb * self.T[i])) - 1
            black_body = freqs**3 / (c.c**2 * wien)
            spectrum += self.kappa_plank[i] * np.exp(-self.tau_eff[i]) * black_body
        
        norm = self.L_photo / np.trapz(spectrum, freqs)
        self.spectrum = spectrum * norm
        
        
#%% Do it
rays = []

for observer, corner, iobs in tqdm(zip(observers, corners, range(0, 192))):
    iobs = 188
    corner_theta = corner[0] # 0.332 / 2 
    corner_phi = corner[1] # 0.339 / 2 
    
    # Make mask
    patrol = borderlands(corner_theta, corner_phi)
    tube_mask = patrol(THETA, PHI)
    fluff_mask = Den > 1e-19
    mask = tube_mask * fluff_mask

    ray_temp = ray(T[mask], Den[mask], R[mask], THETA[mask], PHI[mask], 
                   Rad[mask], X[mask], Y[mask], Z[mask], Vol[mask],
                   observer[0], observer[1]) # theta, phi
    
    ray_temp.sort() # Sort by r
    ray_temp.calc_taus(eng)
    ray_temp.calc_flux()
    ray_temp.calc_colorsphere()
    # ray_temp.calc_spectra(freqs, cross_dot)
    rays.append(ray_temp)
    break
    # Do the cross dot bit
#%%
red = 0
photo = 0
for ray in rays:
    red += ray.L_photo
    photo += ray.R[ray.photosphere] / c.Rsol_to_cm
red /= 192
photo /= 192

# Photosphere save
import csv
# Save photocolor
pre_saving = 'data/photosphere'
filepath =  f'{pre_saving}/tube_photo1.csv'
photos = [r.R[r.photosphere] / c.Rsol_to_cm for r in rays]
photos_dir_X = [r.X[r.photosphere] / c.Rsol_to_cm for r in rays]
photos_dir_Y = [r.Y[r.photosphere] / c.Rsol_to_cm for r in rays]
colors_dir_X = [r.X[r.colorsphere] / c.Rsol_to_cm for r in rays]
colors_dir_Y = [r.Y[r.colorsphere] / c.Rsol_to_cm for r in rays]

with open(filepath, 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(photos)
file.close()

