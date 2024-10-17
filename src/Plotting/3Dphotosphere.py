#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 17:32:36 2024

@author: konstantinos
"""
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import plotly.graph_objects as go
from plotly.offline import plot
from scipy.interpolate import griddata

from src.Utilities.find_cardinal_directions import find_sph_coord
import src.Utilities.prelude as c

rstar = 0.47
mstar = 0.5
Mbh = 100000 #'1e+06'
extra = 'beta1S60n1.5Compton'
simname = f'R{rstar}M{mstar}BH{Mbh}{extra}' 
fix = 302
photo = np.loadtxt(f'data/blue2/{simname}/photosphere{fix}.txt')
therm_rad = np.loadtxt(f'data/blue2/{simname}/thermradius{fix}.txt')
temperatures = np.loadtxt(f'data/blue2/{simname}/temperaturemap{fix}.txt')
spectra = np.loadtxt(f'data/blue2/{simname}/spectra{fix}.txt')
freqs = np.loadtxt(f'data/blue2/{simname}/freqs.txt',)
spectra = np.multiply(spectra, freqs)
Rt = rstar * (float(Mbh)/mstar)**(1/3)
#%% 

#-- Interpolate photosphere
x_hp = [] #np.ones(192)
y_hp = [] #np.ones(192)
z_hp = [] #np.ones(192)
funnel = []
funnel_xyz = []
not_funnel = []
x_ray = []
for i in range(0,192):
    theta, phi = hp.pix2ang(4, i) # theta in [0,pi], phi in [0,2pi]
    if photo[i]<5000:
        x, y, z = find_sph_coord(theta, phi, photo[i])
        x_hp.append(x)
        y_hp.append(y)
        z_hp.append(z)
    if therm_rad[i]<Rt:
        funnel.append((theta - np.pi/2, phi - np.pi))
        x, y, z = find_sph_coord(theta, phi, therm_rad[i])
        funnel_xyz.append((x, y, z))
    else:
        not_funnel.append((theta - np.pi/2, phi - np.pi, i ))
    if spectra[i][446] > 1e40:
        x_ray.append((theta - np.pi/2, phi - np.pi , i))
        
funnel = np.array(funnel).T
not_funnel = np.array(not_funnel).T
funnel_xyz = np.array(funnel_xyz).T
x_ray = np.array(x_ray).T

#%%
grid_x, grid_y = np.meshgrid(np.linspace(min(x_hp), max(x_hp), 100),
                             np.linspace(min(y_hp), max(y_hp), 100))
grid_z = griddata((x_hp, y_hp), z_hp, 
                  (grid_x, grid_y), method='linear')
grid_r = np.sqrt(grid_x**2 + grid_y**2 + grid_z**2)

#-- Tidal Radius
x_rt = np.zeros(192)
y_rt = np.zeros(192)
z_rt = np.zeros(192)
for i in range(0,192):
    theta, phi = hp.pix2ang(4, i) # theta in [0,pi], phi in [0,2pi]
    x_rt[i], y_rt[i], z_rt[i] = find_sph_coord(theta, phi, Rt)
grid_xrt, grid_yrt = np.meshgrid(np.linspace(min(x_rt), max(x_rt), 100),
                             np.linspace(min(y_rt), max(y_rt), 100))
grid_zrt = griddata((x_rt, y_rt), z_rt, 
                  (grid_xrt, grid_yrt), method='linear')

#--- Plot
photosphere_surface =  go.Surface(z = grid_z, x = grid_x, y = grid_y,
                                  surfacecolor = grid_r,
                                  colorscale='Inferno_r')
photosphere_scatter = go.Scatter3d(z=z_hp, y=y_hp, x=x_hp,
                                    marker=dict(color='red', size=5))
tidal_scatter = go.Scatter3d(z=z_rt, y=y_rt, x=x_rt,
                              marker=dict(color='black', size=2))
# funnel_scatter = go.Scatter3d(z=funnel_xyz[0], y=funnel_xyz[1], x=funnel_xyz[2],
#                                     marker=dict(color='blue', size=5))
# fig = go.Figure(data=[ photosphere_surface, 
#                       photosphere_scatter, tidal_s/catter,
                        
#                       ])
# fig.update_layout(title='3D Surface Plot', 
#                   autosize=False,
#                   width=1000, height=1000,
#                   margin=dict(l=65, r=50, b=65, t=90))
# plot(fig, auto_open=True)
# fig.show()
#%%
ax = plt.figure(figsize=(4,4), dpi = 300).add_subplot(111, projection='mollweide')
ax.scatter(not_funnel[1], not_funnel[0], c ='k', marker = 'h', s = 20)
ax.scatter(, x_ray[0], c ='skyblue', marker = 'h', s = 20)
#%%

x_mask = np.zeros(192)
x_ray_obs = []
xray_idx = np.argmin(np.abs(freqs - 1e18))
for i, spectrum in enumerate(spectra):
    theta, phi = hp.pix2ang(4, i) # theta in [0,pi], phi in [0,2pi]
    if spectra[i][xray_idx] > 1e40:
        print('hi')
        x_mask[i] = 1
        x_ray_obs.append((theta - np.pi/2, phi - np.pi))
x_ray_obs = np.array(x_ray_obs).T
ax = plt.figure(figsize=(4,4), dpi = 300).add_subplot(111, projection='mollweide')
ax.scatter(funnel[1], funnel[0], c = 'skyblue', marker = 'h', s = 20)
ax.scatter(not_funnel[1], not_funnel[0], c ='k', marker = 'h', s = 20, )
ax.scatter(x_ray_obs[1], x_ray_obs[0], c = c.AEK, marker = 'h', s = 30, alpha = 0.75)
#%% Therm and photo
plt.figure()
plt.plot(therm_rad, '^', c = 'g', label = 'Therm')
plt.plot(photo, 's', c=c.AEK, label = 'Photo', alpha = 0.5)
plt.axhline(Rt, c = 'k', ls = '--')
plt.text(100, Rt + 10, r'$R_\mathrm{T}$', c='k')
plt.text(100, 0.6*Rt + 5, r'$R_\mathrm{soft}$', c='r')
plt.axhline(0.6 * Rt, c = 'r', ls = '-.')
plt.ylim(5, 10000)
plt.yscale('log')
plt.ylabel('Therm Radius [Rsol]')
plt.xlabel('Observer id')
plt.legend()
#%% How many above?
plt.figure()
plt.plot(photo - therm_rad, c='k')
plt.axhline(0, c='r', ls = '--')