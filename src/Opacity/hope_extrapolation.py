#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: paola, konstantinos

Produce a new table already expanded, in order to interpolate here.
"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.facecolor']= 	'whitesmoke'
import colorcet

##
# VARIABLES 
##

kind = 'rosseland'
save = False
plot = True
plot_linear = False
real_value = False
plot_mesh = False

##
# MAIN
##

# All units are ln[cgs]
loadpath = 'src/Opacity/'
lnT = np.loadtxt(loadpath + 'T.txt')
lnrho = np.loadtxt(loadpath + 'rho.txt')
if kind == 'rosseland':
    lnk = np.loadtxt(loadpath + 'ross.txt')
elif kind == 'planck':
    lnk = np.loadtxt(loadpath + 'planck.txt')
elif kind == 'scatter':
    lnk = np.loadtxt(loadpath + 'scatter.txt')

# Minimum we need is 3.99e-22, Elad's lnrho stops at 1e-10
#we set rho_max = min(lnrho), and use lnrho[1] to make the line. 
# Otherwise there's a jump.
rho_min = np.log(3.99e-20)
delta_rho = lnrho[1] - lnrho[0]
rho_max = lnrho[0] - delta_rho

expanding_rho = np.arange(rho_min,rho_max, delta_rho)
table_expansion = np.zeros( (len(lnT), len(expanding_rho) ))


for i in range(len(lnT)):

    opacity_row = lnk[i]
    fit = np.polyfit(lnrho[0:100], opacity_row[0:100], deg = 1)
    
    for j, rho in enumerate(expanding_rho):
        table_expansion[i, j] = fit[0] * rho + fit[1]
    # if kind == 'planck':s
    # midx = 1
    # sidx = 1
    # delta_opacity = opacity_row[midx] - opacity_row[midx - 1]
    # delta_rho_fix = lnrho[midx] - lnrho[midx -1]
    # m = np.divide(delta_opacity, delta_rho_fix)
    # for j in range(len(expanding_rho)):           
    #     table_expansion[i,j] = opacity_row[sidx] + m * (expanding_rho[j]-lnrho[sidx])
    # else:
    #     delta_opacity = opacity_row[1] - opacity_row[0]
    #     m = np.divide(delta_opacity, delta_rho)
    #     for j in range(len(expanding_rho)-1):           
    #         table_expansion[i,j] = opacity_row[2] + m * (expanding_rho[j]-lnrho[2])

    # expanding_rho[-1] = lnrho[0]
    # table_expansion[i][-1] = opacity_row[0]
# Combine
# lnrho_adjust = np.delete(lnrho,0)
# lnk = np.delete(lnk,0, axis = 1)
ln_new_rho = np.concatenate((expanding_rho, lnrho))
ln_new_table = np.concatenate( (table_expansion, lnk), axis = 1)

if save:
    if kind == 'rosseland':
        np.savetxt(loadpath + 'hope_ross_expansion.txt', ln_new_table)
    elif kind == 'planck':
        np.savetxt(loadpath + 'hope_planck_expansion.txt', ln_new_table)
    elif kind == 'scatter':
        np.savetxt(loadpath + 'hope_scatter_expansion.txt', ln_new_table)
            
    np.savetxt(loadpath + 'hope_big_lnrho.txt', ln_new_rho)

if plot:   
    # Extrapol.
    lnrho = np.loadtxt(loadpath + 'hope_big_lnrho.txt')
    lnk_planck = np.loadtxt(loadpath + 'hope_planck_expansion.txt')
    
    plancks = []
    for i, T in enumerate(lnT):   
        planck = [ np.log10( np.exp(lnk_planck[i, j])) for j, rho in enumerate(lnrho)]
        plancks.append(planck)
        
    oldrho = np.loadtxt(loadpath + 'rho.txt')

    for planck in plancks:
        plt.plot(np.log10(np.exp(lnrho)), planck, c = 'k')           
        
    plt.axvline( np.log10(np.exp(oldrho[0])), c = 'r')
    plt.grid()
    #plt.ylim(-120, 10)
    plt.title( 'Our Extrapolation, Every T')
    plt.xlabel(r'Density $\log_{10}( \rho )$ [g/cm$^3$]')
    plt.ylabel(r'Opacity $\log_{10}(\kappa)$ [1/cm$^{-1}$]')
    plt.show()
    
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
    
