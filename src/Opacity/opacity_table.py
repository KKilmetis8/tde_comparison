#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 16:26:39 2023

@author: konstantinos

NOTES FOR OTHERS
- T, rho are in CGS
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

# All units are ln[cgs]
loadpath = 'src/Opacity/'
lnT = np.loadtxt(loadpath + 'T.txt')
lnrho = np.loadtxt(loadpath + 'big_lnrho.txt')
# lnk_ross = np.loadtxt(loadpath + 'ross.txt')
# lnk_planck = np.loadtxt(loadpath + 'planck.txt')
# lnk_scatter = np.loadtxt(loadpath + 'scatter.txt')
lnk_ross = np.loadtxt(loadpath + 'ross_expansion.txt')
lnk_planck = np.loadtxt(loadpath + 'planck_expansion.txt')
lnk_scatter = np.loadtxt(loadpath + 'scatter_expansion.txt')

lnk_ross_inter = RegularGridInterpolator( (lnT, lnrho), lnk_ross)
lnk_planck_inter = RegularGridInterpolator( (lnT, lnrho), lnk_planck)
lnk_scatter_inter = RegularGridInterpolator( (lnT, lnrho), lnk_scatter)

def opacity(T, rho, kind, ln = True) -> float:
    '''
    Return the rosseland mean opacity in [cgs], given a value of density,
    temperature and and a kind of opacity. If ln = True, then T and rho are
    lnT and lnrho. Otherwise we convert them.
    
     Parameters
     ----------
     T : float,
         Temperature in [cgs].
     rho : float,
         Density in [cgs].
     kind : str,
         The kind of opacities. Valid choices are:
         rosseland, plank or effective.
     log : bool,
         If True, then T and rho are lnT and lnrho, Default is True
     
    Returns
    -------
    opacity : float,
        The rosseland mean opacity in [cgs].
    '''    
    if not ln: 
        T = np.log(T)
        rho = np.log(rho)
        # Remove fuckery
        T = np.nan_to_num(T, nan = 0, posinf = 0, neginf= 0)
        rho = np.nan_to_num(rho, nan = 0, posinf = 0, neginf= 0)

    # Pick Opacity & Use Interpolation Function
    if kind == 'rosseland':
        ln_opacity = lnk_ross_inter((T, rho))
        
    elif kind == 'planck':
        ln_opacity = lnk_planck_inter((T, rho))
        
    elif kind == 'scattering':
        ln_opacity = lnk_scatter_inter((T, rho))
        
    elif kind == 'effective':
        absorption = lnk_ross_inter((T, rho))
        scattering = lnk_scatter_inter((T, rho))
        
        # Apoelenism
        k_a = np.exp(absorption)
        k_s = np.exp(scattering)
        
        # Rybicky & Lightman eq. 1.98 NO USE STEINGERG & STONE (9)
        opacity = np.sqrt(3 * k_a * (k_a + k_s)) 
        return opacity
    
    elif kind == 'red':
        planck = lnk_planck_inter((T, rho))
        scattering = lnk_scatter_inter((T, rho))
        
        # Apoelenism
        k_p = np.exp(planck)
        k_s = np.exp(scattering)
        
        opacity = k_p + k_s
        return opacity
    
    else:
        print('Invalid opacity type. Try: scattering/ rosseland / planck / effective.')
        return 1
    
    opacity = np.exp(ln_opacity)
    
    return opacity

if __name__ == '__main__':
    
    elena = True
    if elena:
        lnT = np.loadtxt(loadpath + 'T.txt')
        lnrho = np.loadtxt(loadpath + 'rho.txt')
        lnk_ross = np.loadtxt(loadpath + 'ross.txt')
        lnk_planck = np.loadtxt(loadpath + 'planck.txt')
        lnk_scatter = np.loadtxt(loadpath + 'scatter.txt')
        
        rho1 = lnrho[0]
        rho2 = lnrho[ len(lnrho)// 2]
        rho3 = lnrho[-1]
        
        scatter1 = [ np.log(opacity(T, rho1, kind = 'scattering', ln = True)) for T in lnT]
        scatter2 = [ np.log(opacity(T, rho2, kind = 'scattering', ln = True)) for T in lnT]
        scatter3 = [ np.log(opacity(T, rho3, kind = 'scattering', ln = True)) for T in lnT]
        
        planck1 = [ np.log(opacity(T, rho1, kind = 'planck', ln = True)) for T in lnT]
        planck2 = [ np.log(opacity(T, rho2, kind = 'planck', ln = True)) for T in lnT]
        planck3 = [ np.log(opacity(T, rho3, kind = 'planck', ln = True)) for T in lnT]

        import matplotlib.pyplot as plt
        plt.rcParams['text.usetex'] = True
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['axes.facecolor']= 	'whitesmoke'
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        AEK = '#F1C410'
        
        fig, ax = plt.subplots(figsize = (5,4))
        plt.plot(np.log10(np.exp(lnT)), scatter1, c = AEK, 
                 label =  rf'Scatter $ \rho $ {np.exp(rho1):.1e}')
        plt.plot(np.log10(np.exp(lnT)), planck1, c = AEK, linestyle = '--', 
                 label =  rf'Planck $ \rho $ {np.exp(rho1):.1e}')
        # 
        plt.plot(np.log10(np.exp(lnT)), scatter2, c = 'k', 
                 label =  rf'Scatter $ \rho $ {np.exp(rho2):.1e}')
        plt.plot(np.log10(np.exp(lnT)), planck2, c = 'k', linestyle = '--', 
                 label =  rf'Planck $ \rho $ {np.exp(rho2):.1e}')
        #
        plt.plot(np.log10(np.exp(lnT)), scatter3, c = 'maroon', 
                 label =  rf'Scatter $ \rho $ {np.exp(rho3):.1e}')
        plt.plot(np.log10(np.exp(lnT)), planck3, c = 'maroon', linestyle = '--', 
                 label =  rf'Planck $ \rho $ {np.exp(rho3):.1e}')
        #
        plt.xlabel(r'Temperature $\log_{10}(T)$ [K]')
        plt.ylabel(r'Opacity $\log_{10}(\kappa)$ [1/cm$^{-1}$]')
        plt.title('Comparing Planck vs Scattering opacities')
        plt.legend( bbox_to_anchor=(1.15,-0.15), ncols = 3,
                   bbox_transform = ax.transAxes )
        plt.grid()
