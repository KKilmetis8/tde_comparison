#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: paola

Produce a new table already expanded, in order to interpolate here.
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.facecolor']= 	'whitesmoke'
import colorcet
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d

##
# VARIABLES 
##

kind = 'planck'
save = False
plot_linear = True
real_value = False
plot_mesh = False

##
# MAIN
##

# All units are ln[cgs]
loadpath = 'src/Opacity/'
lnT = np.loadtxt(loadpath + 'T.txt')
lnrho = np.loadtxt(loadpath + 'rho.txt')

# Minimum we need is 3.99e-22, Elad's lnrho stops at 1e-10
#we set rho_max = min(lnrho), and use lnrho[1] to make the line. 
# Otherwise there's a jump.
rho_min = np.log(3.99e-22)
rho_max = lnrho[0] #np.log(8e-11)
expanding_rho = np.arange(rho_min,rho_max, 0.2)
table_expansion = np.zeros( (len(lnT), len(expanding_rho) ))
delta_rho = lnrho[1] - lnrho[0]

for i in range(len(lnT)):
    if kind == 'rosseland':
        lnk = np.loadtxt(loadpath + 'ross.txt')
    elif kind == 'planck':
        lnk = np.loadtxt(loadpath + 'planck.txt')
    elif kind == 'scatter':
        lnk = np.loadtxt(loadpath + 'scatter.txt')

    opacity_row = lnk[i]
    
    if kind == 'planck':
        delta_opacity = opacity_row[3] - opacity_row[2]
        delta_rho_fix = lnrho[3] - lnrho[2]
        m = np.divide(delta_opacity, delta_rho_fix)
        for j in range(len(expanding_rho)-1):           
            table_expansion[i,j] = opacity_row[0] + m * (expanding_rho[j]-lnrho[0])
    else:
        delta_opacity = opacity_row[1] - opacity_row[0]
        m = np.divide(delta_opacity, delta_rho)
        for j in range(len(expanding_rho)-1):           
            table_expansion[i,j] = opacity_row[0] + m * (expanding_rho[j]-lnrho[0])

    expanding_rho[-1] = lnrho[0]
    table_expansion[i][-1] = opacity_row[0]
# Combine
# lnrho_adjust = np.delete(lnrho,0)
# lnk = np.delete(lnk,0, axis = 1)
ln_new_rho = np.concatenate((expanding_rho, lnrho))
ln_new_table = np.concatenate( (table_expansion, lnk), axis = 1)


if save:
    if kind == 'rosseland':
        np.savetxt(loadpath + 'ross_expansion.txt', ln_new_table)
    elif kind == 'planck':
        np.savetxt(loadpath + 'planck_expansion.txt', ln_new_table)
    elif kind == 'scatter':
        np.savetxt(loadpath + 'scatter_expansion.txt', ln_new_table)
            
    np.savetxt(loadpath + 'big_lnrho.txt', ln_new_rho)

# Plotting
if plot_mesh:
    # Norm
    cmin = -30
    cmax = 20
    
    k = lnk
        
    # Elad's Table
    fig  = plt.figure( figsize = (6,4))
    img = plt.pcolormesh(lnrho, lnT, k, 
                          cmap = 'cet_fire', vmin = cmin, vmax = cmax)

    plt.xlabel(r'$\ln ( \rho )$ $[g/cm^3]$')
    plt.ylabel('$\ln(T)$ $[K]$')
    if kind == 'rosseland':
        plt.title('Rosseland Mean Opacity | Elad Table')
    elif kind == 'planck':
        plt.title('Planck Mean Opacity | Elad Table')
    elif kind == 'scatter':
        plt.title('Scatter Mean Opacity | Elad Table')
        
    cax = fig.add_axes([0.93, 0.125, 0.04, 0.76])
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label('$\ln(\kappa)$ $[cm^-1]$', rotation=270, labelpad = 15)
    
    # Extrapolated Table
    fig = plt.figure( figsize = (8,4) )
    img = plt.pcolormesh(ln_new_rho, lnT, ln_new_table, 
                          cmap = 'cet_fire', vmin = cmin, vmax = cmax)
    plt.xlabel(r'$\ln( \rho )$ $[g/cm^3]$')
    plt.ylabel('$\ln(T)$ $[K]$')
    
    plt.axvline( (expanding_rho[-1] + lnrho[0]) /2 , 
                color = 'b', linestyle = 'dashed')
    
    cax = fig.add_axes([0.92, 0.125, 0.03, 0.76])
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label('$\ln(\kappa)$ $[cm^{-1}]$', rotation=270, labelpad = 15)
    if kind == 'rosseland':
        plt.title('Rosseland Mean Opacity | Extrapolated Table')
        plt.savefig('Figs/rosseland.png')
    elif kind == 'planck':
        plt.title('Planck Mean Opacity | Extrapolated Table')
        plt.savefig('Figs/planck.png')
    elif kind == 'scatter':
        plt.title('Scatter Mean Opacity | Extrapolated Table')
        plt.savefig('Figs/scatter.png')
    plt.show()
    

if plot_linear:   
    ln_rho_old = np.loadtxt('OLD stuff/OLDbig_lnrho.txt')
    if kind == 'rosseland':
        ln_old = np.loadtxt('OLD stuff/OLDross_expansion.txt')
    elif kind == 'planck':
        ln_old = np.loadtxt('OLD stuff/OLDplanck_expansion.txt')
    elif kind == 'scatter':
        ln_old = np.loadtxt('OLD stuff/OLDscatter_expansion.txt')
    
    if real_value:
        rho_old = np.exp(ln_rho_old)
        rho = np.exp(lnrho)
        old = np.exp(ln_old)  
        k = np.exp(lnk)     
        new_rho = np.exp(ln_new_rho)
        new_table = np.exp(ln_new_table) 

        plt.plot(new_rho, new_table[i, :], label = 'HOPE extrapolation', c = 'orange')
        plt.plot(rho_old, old[i, :], label = 'OLD extrapolation', linestyle = '--', c ='g')
        plt.scatter(rho, k[i, :], c = 'r', s = 2, label = 'Elad')
        plt.axvline(rho[0], c = 'black', linestyle = '--')
        #plt.text(-50, 25, 'T:  {:.2f}'.format(np.exp(lnT[i])) + '\n position: ' + str(i))
        plt.xlabel(r'$\rho$ [g/cm$^2$]')
        plt.ylabel('K [1/cm]')
        #plt.ylim(-50,25)
        #plt.legend()
        if kind == 'rosseland':
            plt.title('Rosseland Mean Opacity')
            #plt.savefig('Figs/rosseland_linear.png')
            #plt.savefig('Figs/rosseland_linear.png')
        elif kind == 'planck':
            plt.title('Planck Mean Opacity')
        elif kind == 'scatter':
            plt.title('Scatter Mean Opacity')
        plt.show()

    else:
        for i in range(62,len(lnT)):
            fig  = plt.figure(figsize = (4,3))  
            plt.plot(ln_new_rho, ln_new_table[i, :], label = 'HOPE extrapolation', c = 'orange')
            plt.plot(ln_rho_old, ln_old[i, :], label = 'OLD extrapolation', linestyle = '--', c ='g')
            plt.scatter(lnrho, lnk[i, :], c = 'r', s = 2, label = 'Elad')
            plt.axvline(lnrho[0], c = 'black', linestyle = '--')
            plt.text(-50, 20, 'T:  {:.2f}'.format(np.exp(lnT[i])) + '\n position: ' + str(i))
            plt.ylim(-100,30)
            plt.xlabel(r'$\ln\rho$ [g/cm$^2$]')
            plt.ylabel(r'$\ln$ K [1/cm]')
            #plt.legend()
            if kind == 'rosseland':
                plt.title('Rosseland Mean Opacity')
                #plt.savefig('Figs/rosseland_linear.png')
            elif kind == 'planck':
                plt.title('Planck Mean Opacity')
            elif kind == 'scatter':
                plt.title('Scatter Mean Opacity')
            plt.show()
