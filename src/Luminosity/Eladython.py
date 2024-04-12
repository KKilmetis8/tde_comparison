#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 18:52:21 2024

@author: konstantinos
"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

# The goal is to replicate Elad's script. No nice code, no nothing. A shit ton
# of comments though. 
import numpy as np
import h5py
import matplotlib.pyplot as plt

import src.Utilities.prelude as c
# Okay, import the constants. Do not be absolutely terrible'''
    # n_start = snap_no_start
    # n_end = snap_no_end
# Since this first for loop ends with the end of the code, I think it iterates
# over every snapshot. This is our choose_snap code
#%% Load data -----------------------------------------------------------------
pre = '5/'
fix = '308'
X = np.load(pre + fix + '/CMx_' + fix + '.npy')
Y = np.load(pre + fix + '/CMy_' + fix + '.npy')
Z = np.load(pre + fix + '/CMz_' + fix + '.npy')
VX = np.load(pre + fix + '/Vx_' + fix + '.npy')
VY = np.load(pre + fix + '/Vy_' + fix + '.npy')
VZ = np.load(pre + fix + '/Vz_' + fix + '.npy')
T = np.load(pre + fix + '/T_' + fix + '.npy')
Den = np.load(pre + fix + '/Den_' + fix + '.npy')
Rad = np.load(pre + fix + '/Rad_' + fix + '.npy')
IE = np.load(pre + fix + '/IE_' + fix + '.npy')
Vol = np.load(pre + fix + '/Vol_' + fix + '.npy')

box = np.zeros(6)
with h5py.File(f'{pre}{fix}/snap_{fix}.h5', 'r') as fileh:
    for i in range(len(box)):
        box[i] = fileh['Box'][i]
    fileh.close()
        
Rad_den = np.multiply(Rad,Den)
R = np.sqrt(X**2 + Y**2 + Z**2)
velocity_vector = np.divide( VX * X + VY * Y + VZ * Z,R)
velocity = np.sqrt(VX**2 + VY**2 + VZ**2)
#%% Cross dot -----------------------------------------------------------------
import healpy as hp
observers_xyz = hp.pix2vec(c.NSIDE, range(192))
# Line 17, * is matrix multiplication, ' is .T
observers_xyz = np.array(observers_xyz).T
cross_dot = np.matmul(observers_xyz,  observers_xyz.T)
cross_dot[cross_dot<0] = 0
cross_dot *= 4/192

#%% Opacities -----------------------------------------------------------------
# Freq range
f_min = c.Kb * 1e3 / c.h
f_max = c.Kb * 3e13 / c.h
f_num = 1_000
frequencies = np.logspace(np.log10(f_min), np.log10(f_max), f_num)

# Opacity Input
opac_kind = 'LTE'
opac_path = f'src/Opacity/{opac_kind}_data/'
T_cool = np.loadtxt(f'{opac_path}/T.txt')
Rho_cool = np.loadtxt(f'{opac_path}/rho.txt')
plank = np.loadtxt(f'{opac_path}/planck.txt')
rossland = np.loadtxt(f'{opac_path}/ross.txt')

# Opacity Interpolation
# T_interp, Rho_interp = np.meshgrid(T_cool,Rho_cool) all commented out
# rossland = np.reshape(rossland, (len(T_cool), len(Rho_cool)))
# plank = np.reshape(plank, (len(T_cool), len(Rho_cool)))

# Fill value none extrapolates
def linearpad(D0,z0):
    factor = 100
    dz = z0[-1] - z0[-2]
    # print(np.shape(D0))
    dD = D0[:,-1] - D0[:,-2]
    
    z = [zi for zi in z0]
    z.append(z[-1] + factor*dz)
    
    z = np.array(z)
    
    #D = [di for di in D0]

    to_stack = np.add(D0[:,-1], factor*dD)
    to_stack = np.reshape(to_stack, (len(to_stack),1) )
    D = np.hstack((D0, to_stack))
    #D.append(to_stack)
    return np.array(D), z

def pad_interp(x,y,V):
    Vn, xn = linearpad(V, x)
    Vn, xn = linearpad(np.fliplr(Vn), np.flip(xn))
    Vn = Vn.T
    Vn, yn = linearpad(Vn, y)
    Vn, yn = linearpad(np.fliplr(Vn), np.flip(yn))
    Vn = Vn.T
    return xn, yn, Vn
# #%%    
T_cool2, Rho_cool2, rossland2 = pad_interp(T_cool, Rho_cool, rossland.T)
_, _, plank2 = pad_interp(T_cool, Rho_cool, plank.T)

#%% Tree ----------------------------------------------------------------------
import matlab.engine
eng = matlab.engine.start_matlab()
#from scipy.spatial import KDTree
xyz = np.array([X, Y, Z]).T
# tree = KDTree(xyz, leafsize=50)
tree = eng.KDTreeSearcher(xyz)
N_ray = 5_000

# In the future these will be ray class

# Holders Rays
# d_ray = np.zeros((N_ray, 192))
# r_ray = np.zeros((N_ray, 192))
# T_ray = np.zeros((N_ray, 192))
# Trad_ray = np.zeros((N_ray, 192))
# Tnew_ray = np.zeros((N_ray, 192))
# tau_ray = np.zeros((N_ray, 192))
# tau_therm_ray = np.zeros((N_ray, 192))
# v_ray = np.zeros((N_ray, 192))
# c_ray = np.zeros((N_ray, 192))

# Holders Photosphere / Thermalization
# d_photo = np.zeros(192)
# x_therm = np.zeros(192) 
# y_therm = np.zeros(192)
# z_therm = np.zeros(192)
# vr_therm = np.zeros(192)
# T_photo = np.zeros(192)
# r_photo = np.zeros(192)
# r_therm = np.zeros(192) 
# photo_ind = np.zeros(192)
# v_photo = np.zeros(192)
# T_avg_photo = np.zeros(192)
# L_avg = np.zeros(192)
# L = np.zeros(192)
# Lnew = np.zeros(192)
# L_g = np.zeros(192)
# L_r = np.zeros(192)
# L_i = np.zeros(192)
# L_uvw2 = np.zeros(192)
# L_uvm2 = np.zeros(192)
# L_uvw1 = np.zeros(192)
# L_uvotu = np.zeros(192)
# L_XRT = np.zeros(192)
# L_XRT2 = np.zeros(192)

# Flux?
F_photo = np.zeros((192, f_num))
F_photo_temp = np.zeros((192, f_num))

# Lines 99-128 use some files we don't have, I think we only need
# these for blue. Ignore for now 

# Dynamic Box -----------------------------------------------------------------
#for i in range(1):
i = 74
# Progress 
if i % 10 == 0:
    print('Eladython Ray no:', i)

mu_x = observers_xyz[i][0]
mu_y = observers_xyz[i][1]
mu_z = observers_xyz[i][2]

# Box is for dynamic ray making
if mu_x < 0:
    rmax = box[0] / mu_x
    # print('x-', rmax)
else:
    rmax = box[3] / mu_x
    # print('x+', rmax)
if mu_y < 0:
    rmax = min(rmax, box[1] / mu_y)
    # print('y-', rmax)
else:
    rmax = min(rmax, box[4] / mu_y)
    # print('y+', rmax)

if mu_z < 0:
    rmax = min(rmax, box[2] / mu_z)
    # print('z-', rmax)
else:
    rmax = min(rmax, box[5] / mu_z)
    # print('z+', rmax)
print(rmax)
# This naming is so bad
r = np.logspace( -0.25, np.log10(rmax), N_ray)
alpha = (r[1] - r[0]) / (0.5 * ( r[0] + r[1]))
dr = alpha * r

x = r*mu_x
y = r*mu_y
z = r*mu_z
xyz2 = np.array([x, y, z]).T
# _ , idx = tree.query(xyz2)
idx = eng.knnsearch(tree, xyz2)
idx = [ int(idx[i][0] - 1) for i in range(len(idx))] # -1 because we start from 0
d = Den[idx] * c.den_converter
t = T[idx]

#%% Interpolate
from scipy.interpolate import griddata
# T_cool2, Rho_cool2 = np.meshgrid(T_cool2,Rho_cool2)
# T_cool2 = T_cool2.ravel()

# Rho_cool2 = Rho_cool2.ravel()

# cool = np.array([T_cool2, Rho_cool2]).T
# sigma_rossland = griddata( cool, rossland2.ravel(),
#                             xi = np.array([np.log(t), np.log(d)]).T )
# sigma_plank = griddata( (T_cool2, Rho_cool2), plank2.ravel(),
#                             xi = np.array([np.log(t), np.log(d)]).T )

# sigma_rossland_eval = np.exp(sigma_rossland)
# sigma_plank_eval = np.exp(sigma_plank)

sigma_rossland = eng.interp2(T_cool2,Rho_cool2,rossland2,np.log(t),np.log(d),'linear',0)
sigma_rossland = [sigma_rossland[0][i] for i in range(N_ray)]
sigma_rossland_eval = np.exp(sigma_rossland) 

sigma_plank = eng.interp2(T_cool2,Rho_cool2,plank2,np.log(t),np.log(d),'linear',0)
sigma_plank = [sigma_plank[0][i] for i in range(N_ray)]
sigma_plank_eval = np.exp(sigma_plank)
#%% Optical Depth ---------------------------------------------------------------
# Okay, line 232, this is the hard one.
import scipy.integrate as sci

r_fuT = np.flipud(r.T)

kappa_rossland = np.flipud(sigma_rossland_eval) 
los = - np.flipud(sci.cumulative_trapezoid(kappa_rossland, r_fuT, initial = 0)) * c.Rsol_to_cm # dont know what it do but this is the conversion

kappa_plank = np.flipud(sigma_plank_eval) 
los_abs = - np.flipud(sci.cumulative_trapezoid(kappa_plank, r_fuT, initial = 0)) * c.Rsol_to_cm

k_effective = np.sqrt(3 * np.flipud(sigma_plank_eval) * np.flipud(sigma_rossland_eval)) 
los_effective = - np.flipud(sci.cumulative_trapezoid(k_effective, r_fuT, initial = 0)) * c.Rsol_to_cm
# tau_tot = dr.T * c.Rsol_to_cm * sigma_rossland_eval

#%% Red -----------------------------------------------------------------------

# Get 20 unique, nearest neighbors
xyz3 = np.array([X[idx], Y[idx], Z[idx]]).T
# _, idxnew = tree.query(xyz3, k=20) # 20 nearest neighbors
idxnew = eng.knnsearch(tree, xyz3, 'K', 20)
#

#idxnew = np.array([idxnew], dtype = int).T #np.reshape(idxnew, (1, len(idxnew))) #.T
idxnew = np.unique(idxnew).T
idxnew = [ int(idxnew[i] -1) for i in range(len(idxnew))]
# Cell radius
dx = 0.5 * Vol[idx]**(1/3)

# Get the Grads
# sphere and get the gradient on them. Is it neccecery to re-interpolate?

# scattered interpolant returns a function
# griddata DEMANDS that you pass it the values you want to eval at
f_inter_input = np.array([ X[idxnew], Y[idxnew], Z[idxnew] ]).T

gradx_p = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                    xi = np.array([ X[idx]+dx, Y[idx], Z[idx]]).T )
gradx_m = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                    xi = np.array([ X[idx]-dx, Y[idx], Z[idx]]).T )
gradx = (gradx_p - gradx_m)/ (2*dx)

gradx = np.nan_to_num(gradx, nan =  0)
grady_p = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                    xi = np.array([ X[idx], Y[idx]+dx, Z[idx]]).T )
grady_m = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                    xi = np.array([ X[idx], Y[idx]-dx, Z[idx]]).T )
grady = (grady_p - grady_m)/ (2*dx)

grady = np.nan_to_num(grady, nan =  0)

gradz_p = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                    xi = np.array([ X[idx], Y[idx], Z[idx]+dx]).T )
gradz_m = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                    xi = np.array([ X[idx], Y[idx], Z[idx]-dx]).T )
# some nans here
gradz_m = np.nan_to_num(gradz_m, nan =  0)
gradz = (gradz_p - gradz_m)/ (2*dx)

grad = np.sqrt(gradx**2 + grady**2 + gradz**2)
gradr = (mu_x * gradx) + (mu_y*grady) + (mu_z*gradz)
# v_grad = np.sqrt( (VX[idx] * gradx)**2 +  (VY[idx] * grady)**2 + (VZ[idx] * gradz)**2)
R_lamda = grad / ( c.Rsol_to_cm * sigma_rossland_eval* Rad_den[idx])
R_lamda[R_lamda < 1e-10] = 1e-10
fld_factor = 3 * (1/np.tanh(R_lamda) - 1/R_lamda) / R_lamda 

eng.exit()

from scipy.ndimage import uniform_filter1d # does moving mean without fucking the shape up
smoothed_flux = -uniform_filter1d(r.T**2 * fld_factor * gradr / sigma_rossland_eval, 7) # i have removed the minus
#%% Spectra
F_photo_temp = np.zeros((192, f_num))
b = np.where( ((smoothed_flux>0) & (los<2/3) ))[0][0]
# elad_b = 3117
# b = elad_b
Lphoto2 = 4*np.pi*c.c*smoothed_flux[b] * c.Msol_to_g / (c.t**2)
EEr = Rad_den[idx]
if Lphoto2 < 0:
    Lphoto2 = 1e100 # it means that it will always pick max_length for the negatives, maybe this is what we are getting wrong
max_length = 4*np.pi*c.c*EEr[b]*r[b]**2 * c.Msol_to_g * c.Rsol_to_cm / (c.t**2)
Lphoto = np.min( [Lphoto2, max_length])
# Lphoto = Lphoto2
# Spectra ---------------------------------------------------------------------
los_effective[los_effective>30] = 30
b2 = np.argmin(np.abs(los_effective-5))
# elad_b2 = 3460
# b2 = elad_b2
for k in range(b2, len(r)):
    wien = np.exp(c.h * frequencies / (c.Kb * t[k])) - 1
    black_body = frequencies**3 / (c.c**2 * wien)
    F_photo_temp[i,:] += sigma_plank_eval[k] * np.exp(-los_effective[k]) * black_body

norm = Lphoto / np.trapz(F_photo_temp[i,:], frequencies)
F_photo_temp[i,:] *= norm
F_photo[i,:] = np.matmul(cross_dot[i,:], F_photo_temp[:,:]) 

#%% Compare -------------------------------------------
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [4 , 4]
plt.rc('xtick', labelsize = 15) 
plt.rc('ytick', labelsize = 15) 
plt.figure()

import mat73
mat = mat73.loadmat('data/data_308.mat')
def temperature(n):
        return n * c.h / c.Kb
elad_T = np.array([ temperature(n) for n in mat['nu']])
for obs in range(1):
    y = np.multiply(mat['nu'], mat['F_photo_temp'][obs])
    plt.loglog(elad_T, y, c='r', ls = ' ', label ='Elad', zorder = 4,
               marker = 'o', markersize = 2)
    
# us
y_us = F_photo_temp[i] * frequencies
# temp_us = [temperature(n) for n in frequencies]
plt.loglog(elad_T, np.abs(y_us), c = 'k',label='us',
           ls = ' ', marker = 'o', markersize = 5)

# pretty
x_start = 1e3
x_end = 1e8
y_lowlim = 1e35#2e39
y_highlim = 1.3e43
plt.xlim(x_start,x_end)
plt.ylim(y_lowlim, y_highlim)
plt.loglog()
plt.grid()
plt.legend(fontsize = 14)
plt.title(r'Spectrum 10$^5$ $M_\odot$, Snap: 308, Observer , no cross dot')
plt.xlabel('Temperature [K]', fontsize = 16)
plt.ylabel(r'$\nu L_\nu$ [erg/s]', fontsize = 16)
plt.savefig(f'spectra{fix}nocrossdot.png')
plt.show()