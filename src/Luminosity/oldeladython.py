#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 18:52:21 2024

@author: konstantinos
"""

# The goal is to replicate Elad's script. No nice code, no nothing. A shit ton
# of comments though. 
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.Opacity.LTE_loader import T_opac_ex, Rho_opac_ex, rossland_ex, plank_ex
import src.Utilities.prelude as c
#%% Load data -----------------------------------------------------------------
pre = '4/'
fix = '272'
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
box = np.load(pre + fix + '/box_' + fix + '.npy')

Rad_den = np.multiply(Rad,Den)
R = np.sqrt(X**2 + Y**2 + Z**2)

#%% Cross dot -----------------------------------------------------------------
import healpy as hp
observers_xyz = hp.pix2vec(c.NSIDE, range(192))
observers_xyz = np.array([observers_xyz])
observers_xyz = np.reshape(observers_xyz, (192,3))
#Line 17, * is matrix multiplication, ' is .T
cross_dot = np.matmul(observers_xyz,  observers_xyz.T )
cross_dot[cross_dot<0] = 0
cross_dot *= 4/192

# Correction!
npix = hp.nside2npix(c.NSIDE)
theta, phi = hp.pix2ang(c.NSIDE, range(npix))
outx = np.sin(theta) * np.cos(phi)
outy = np.sin(theta) * np.sin(phi)
outz = np.cos(theta)
outX = np.array([outx, outy, outz]).T
cross_dot2 = np.dot(outX,  outX.T )
cross_dot2[cross_dot2<0] = 0
cross_dot2 *= 4/192

cross_dot = cross_dot2
#%% Opacities -----------------------------------------------------------------
# Freq range
f_min = c.kb * 1e3 / c.h
f_max = c.kb * 3e13 / c.h
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

# Fill value none extrapolates
def linearpad(D0,z0):
    factor = 100
    dz = z0[-1] - z0[-2]
    # print(np.shape(D0))
    dD = D0[:,-1] - D0[:,-2]
    
    z = [zi for zi in z0]
    z.append(z[-1] + factor*dz)
    
    z = np.array(z)
    
    to_stack = np.add(D0[:,-1], factor*dD)
    to_stack = np.reshape(to_stack, (len(to_stack),1) )
    D = np.hstack((D0, to_stack))
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
# T_cool2, Rho_cool2, rossland2 = pad_interp(T_cool, Rho_cool, rossland.T)
# _, _, plank2 = pad_interp(T_cool, Rho_cool, plank.T)
T_cool2 = T_opac_ex
Rho_cool2 = Rho_opac_ex
rossland2 = rossland_ex
plank2 = plank_ex


#%% Tree ----------------------------------------------------------------------
import matlab.engine
eng = matlab.engine.start_matlab()
# from scipy.spatial import KDTree
from sklearn.neighbors import KDTree
xyz = np.array([X, Y, Z]).T
# ------- bulshit zone begins
# tree = KDTree(xyz, leafsize=50)
# tree = eng.KDTreeSearcher(xyz)
# ---- bulshit zone ends

N_ray = 5_000

# Flux?
F_photo = np.zeros((192, f_num))
F_photo_temp = np.zeros((192, f_num))

# Lines 99-128 use some files we don't have, I think we only need
# these for blue. Ignore for now 
photos_eladython = []
x_eladython = []
y_eladython = []
xc_eladython = []
yc_eladython = []
# Dynamic Box -----------------------------------------------------------------
for i in tqdm(range(192)): #range(88,104)):
    mu_x = outx[i]
    mu_y = outy[i]
    mu_z = outz[i]
    
    # Box is for dynamic ray making
    if mu_x < 0:
        rmax = box[0] / mu_x
    else:
        rmax = box[3] / mu_x
    if mu_y < 0:
        rmax = min(rmax, box[1] / mu_y)
    else:
        rmax = min(rmax, box[4] / mu_y)
    
    if mu_z < 0:
        rmax = min(rmax, box[2] / mu_z)
    else:
        rmax = min(rmax, box[5] / mu_z)
    # print(rmax)
    # This naming is so bad
    r = np.logspace( -0.25, np.log10(rmax), N_ray)
    alpha = (r[1] - r[0]) / (0.5 * ( r[0] + r[1]))
    dr = alpha * r
    
    x = r*mu_x
    y = r*mu_y
    z = r*mu_z
    xyz2 = np.array([x, y, z]).T
    
    # ------- bulshit zone begins
    # idx = eng.knnsearch(tree, xyz2, 'LeafSize', 50)
    # idx = [ int(idx[i][0] - 1) for i in range(len(idx))] # -1 because we start from 0
    tree = KDTree(xyz, leaf_size=50)
    _, idx = tree.query(xyz2, k=1)
    idx = [ int(idx[i][0]) for i in range(len(idx))] # no -1 because we start from 0
    # ---- bulshit zone ends
    
    d = Den[idx] * c.den_converter
    t = T[idx]
    ray_x = X[idx]
    ray_y = Y[idx]
    
    # Interpolate ---
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
    
    sigma_rossland = eng.interp2(T_cool2,Rho_cool2,rossland2.T # needs T for paola
                                 ,np.log(t), np.log(d),'linear',0)
    sigma_rossland = np.array(sigma_rossland)[0]
    sigma_rossland_eval = np.exp(sigma_rossland) 
    
    sigma_plank = eng.interp2(T_cool2,Rho_cool2,plank2.T, 
                              np.log(t),np.log(d),'linear',0)
    sigma_plank = np.array(sigma_plank)[0]
    sigma_plank_eval = np.exp(sigma_plank)
    # Optical Depth ---.
    import scipy.integrate as sci
    
    r_fuT = np.flipud(r.T)
    
    kappa_rossland = np.flipud(sigma_rossland_eval) 
    los = - np.flipud(sci.cumulative_trapezoid(kappa_rossland, r_fuT, initial = 0)) * c.Rsol_to_cm # dont know what it do but this is the conversion
    
    kappa_plank = np.flipud(sigma_plank_eval) 
    los_abs = - np.flipud(sci.cumulative_trapezoid(kappa_plank, r_fuT, initial = 0)) * c.Rsol_to_cm
    
    k_effective = np.sqrt(3 * np.flipud(sigma_plank_eval) * np.flipud(sigma_rossland_eval)) 
    los_effective = - np.flipud(sci.cumulative_trapezoid(k_effective, r_fuT, initial = 0)) * c.Rsol_to_cm
    # tau_tot = dr.T * c.Rsol_to_cm * sigma_rossland_eval
    
    # Red ---
    
    # Get 20 unique, nearest neighbors
    xyz3 = np.array([X[idx], Y[idx], Z[idx]]).T
    # ---- bulshit zone begins
    # _, idxnew = tree.query(xyz3, k=20) # 20 nearest neighbors
    # idxnew = eng.knnsearch(tree, xyz3, 'K', 20)
    #
    
    #idxnew = np.array([idxnew], dtype = int).T #np.reshape(idxnew, (1, len(idxnew))) #.T
    # idxnew = np.unique(idxnew).T
    # idxnew = [ int(idxnew[i] -1) for i in range(len(idxnew))]
    xyz3 = np.array([X[idx], Y[idx], Z[idx]]).T
    _, idxnew = tree.query(xyz3, k=20)
    idxnew = np.unique(idxnew).T
    # ---- bulshit zone ends

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
    #plt.plot(gradx)
    gradx = np.nan_to_num(gradx, nan =  0)
    grady_p = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                        xi = np.array([ X[idx], Y[idx]+dx, Z[idx]]).T )
    grady_m = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                        xi = np.array([ X[idx], Y[idx]-dx, Z[idx]]).T )
    grady = (grady_p - grady_m)/ (2*dx)
    #plt.plot(grady)
    grady = np.nan_to_num(grady, nan =  0)
    
    gradz_p = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                        xi = np.array([ X[idx], Y[idx], Z[idx]+dx]).T )
    gradz_m = griddata( f_inter_input, Rad_den[idxnew], method = 'linear',
                        xi = np.array([ X[idx], Y[idx], Z[idx]-dx]).T )
    # some nans here
    gradz_m = np.nan_to_num(gradz_m, nan =  0)
    gradz = (gradz_p - gradz_m)/ (2*dx)
    #plt.plot(gradz)
    grad = np.sqrt(gradx**2 + grady**2 + gradz**2)
    # print(grad)
    gradr = (mu_x * gradx) + (mu_y*grady) + (mu_z*gradz)
    # v_grad = np.sqrt( (VX[idx] * gradx)**2 +  (VY[idx] * grady)**2 + (VZ[idx] * gradz)**2)
    R_lamda = grad / ( c.Rsol_to_cm * sigma_rossland_eval* Rad_den[idx])
    R_lamda[R_lamda < 1e-10] = 1e-10
    fld_factor = 3 * (1/np.tanh(R_lamda) - 1/R_lamda) / R_lamda 
    from scipy.ndimage import uniform_filter1d # does moving mean without fucking the shape up
    smoothed_flux = -uniform_filter1d(r.T**2 * fld_factor * gradr / sigma_rossland_eval, 7) 
    # Spectra
    try:
        b = np.where( ((smoothed_flux>0) & (los<2/3) ))[0][0]
    except:
        print(f'\n {i}, elad b')
        b = 3117
    photos_eladython.append(r[b])
    x_eladython.append(ray_x[b])
    y_eladython.append(ray_y[b])
    los_effective[los_effective>30] = 30
    b2 = np.argmin(np.abs(los_effective-5))
    
    xc_eladython.append(ray_x[b2])
    yc_eladython.append(ray_y[b2])
    # # elad_b = 3117
    # # b = elad_b
    # Lphoto2 = 4*np.pi*c.c*smoothed_flux[b] * c.Msol_to_g / (c.t**2)
    # EEr = Rad_den[idx]
    # if Lphoto2 < 0:
    #     Lphoto2 = 1e100 # it means that it will always pick max_length for the negatives, maybe this is what we are getting wrong
    # max_length = 4*np.pi*c.c*EEr[b]*r[b]**2 * c.Msol_to_g * c.Rsol_to_cm / (c.t**2)
    # Lphoto = np.min( [Lphoto2, max_length])
    # # Lphoto = Lphoto2
    # # Spectra ---------------------------------------------------------------------
    # los_effective[los_effective>30] = 30
    # b2 = np.argmin(np.abs(los_effective-5))
    # # elad_b2 = 3460
    # # b2 = elad_b2
    # for k in range(b2, len(r)):
    #     wien = np.exp(c.h * frequencies / (c.kb * t[k])) - 1
    #     black_body = frequencies**3 / (c.c**2 * wien)
    #     F_photo_temp[i,:] += sigma_plank_eval[k] * np.exp(-los_effective[k]) * black_body
    
    # norm = Lphoto / np.trapz(F_photo_temp[i,:], frequencies)
    # F_photo_temp[i,:] *= norm
    # F_photo[i,:] = np.dot(cross_dot[i,:], F_photo_temp)      


#%% Compare -------------------------------------------
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [4 , 4]
plt.rc('xtick', labelsize = 15) 
plt.rc('ytick', labelsize = 15) 
plt.figure()

import mat73
mat = mat73.loadmat('data/data_308.mat')
def temperature(n):
        return n * c.h / c.kb
elad_T = np.array([ temperature(n) for n in mat['nu']])
for obs in range(1):
    y = np.multiply(mat['nu'], mat['F_photo'][obs])
    # plt.loglog(elad_T, y, c='r', ls = ' ', label ='Elad', zorder = 4,
    #            marker = 'o', markersize = 2)
    
# us
y_us = F_photo[i] * frequencies
# temp_us = [temperature(n) for n in frequencies]
# plt.loglog(elad_T, np.abs(y_us), c = 'k',label='us',
#            ls = ' ', marker = 'o', markersize = 5)
plt.loglog(r, los_effective, c = 'k', lw = 3)
plt.loglog(r, mat['tau_therm_ray'].T[obs], c = 'r')
plt.axhline(5, ls = '--')
plt.axvline(r[b2], c = 'grey', ls = '-.', lw = 1)
plt.axvline(r[3460], c = 'b', ls = '-.', lw = 1)
plt.axvline(mat['r_therm'][obs], c = 'maroon', ls = '-.')
plt.ylim(1e-2, 1e2)
plt.ylabel('Optical Depth (eff)')
# pretty
# x_start = 1e3
# x_end = 1e8
# y_lowlim = 1e35#2e39
# y_highlim = 1.3e43
# plt.xlim(x_start,x_end)
# plt.ylim(y_lowlim, y_highlim)
# plt.loglog()
# plt.grid()
# plt.legend(fontsize = 14)
# plt.title(r'Spectrum 10$^5$ $M_\odot$, Snap: 308, Observer , with cross dot')
# plt.xlabel('Temperature [K]', fontsize = 16)
# plt.ylabel(r'$\nu L_\nu$ [erg/s]', fontsize = 16)