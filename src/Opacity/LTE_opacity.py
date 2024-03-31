#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 16:26:39 2023

@author: konstantinos

NOTES FOR OTHERS
- T, rho are in CGS
"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import src.Utilities.prelude as c

# All units are ln[cgs]
loadpath = 'src/Opacity/LTE_data/'
lnT = np.loadtxt(loadpath + 'T.txt')
lnrho = np.loadtxt(loadpath + 'hope_big_lnrho.txt')
# lnk_ross = np.loadtxt(loadpath + 'ross.txt')
# lnk_planck = np.loadtxt(loadpath + 'planck.txt')
# lnk_scatter = np.loadtxt(loadpath + 'scatter.txt')
lnk_ross = np.loadtxt(loadpath + 'hope_ross_expansion.txt')
lnk_planck = np.loadtxt(loadpath + 'hope_planck_expansion.txt')
lnk_scatter = np.loadtxt(loadpath + 'hope_scatter_expansion.txt')

lnk_ross_inter = RegularGridInterpolator( (lnT, lnrho), lnk_ross)
lnk_planck_inter = RegularGridInterpolator( (lnT, lnrho), lnk_planck)
lnk_scatter_inter = RegularGridInterpolator( (lnT, lnrho), lnk_scatter)

def opacity(T, rho, kind, ln = False) -> float:
    '''
    Return the rosseland mean opacity in [cgs], given a value of density,
    temperature and a kind of opacity. If ln = True, then T and rho are
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
        absorption = lnk_planck_inter((T, rho))
        scattering = lnk_scatter_inter((T, rho))
        
        # Apoelenism
        k_a = np.exp(absorption)
        k_s = np.exp(scattering)
        
        # STEINBERG & STONE (9) (Rybicky & Lightman eq. 1.98)
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
        print('Invalid opacity type. Try: scattering/ rosseland / planck / effective / red.')
    
    opacity = np.exp(ln_opacity)
    
    return opacity

if __name__ == '__main__':
    
    elena = False
    extrapolation_comp = False
    test_elad = True

    if elena:
        lnT = np.loadtxt(loadpath + 'T.txt')
        lnrho = np.loadtxt(loadpath + 'rho.txt')
        lnbigrho = np.loadtxt(loadpath + 'big_lnrho.txt')
        
        rho0 =  lnbigrho[100]
        rho1 = lnrho[0]
        rho2 = lnrho[ len(lnrho)// 2]
        rho3 = lnrho[-1]
        
        scatter0 = [ np.log(opacity(T, rho0, kind = 'scattering', ln = True)) for T in lnT]
        scatter1 = [ np.log(opacity(T, rho1, kind = 'scattering', ln = True)) for T in lnT]
        scatter2 = [ np.log(opacity(T, rho2, kind = 'scattering', ln = True)) for T in lnT]
        scatter3 = [ np.log(opacity(T, rho3, kind = 'scattering', ln = True)) for T in lnT]
        
        planck0 = [ np.log(opacity(T, rho0, kind = 'planck', ln = True)) for T in lnT]
        planck1 = [ np.log(opacity(T, rho1, kind = 'planck', ln = True)) for T in lnT]
        planck2 = [ np.log(opacity(T, rho2, kind = 'planck', ln = True)) for T in lnT]
        planck3 = [ np.log(opacity(T, rho3, kind = 'planck', ln = True)) for T in lnT]
        
        fig, ax = plt.subplots(figsize = (5,4))
        plt.plot(np.log10(np.exp(lnT)), scatter0, c = 'b', 
                 label =  rf'Scatter $ \rho $ {np.exp(rho0):.1e}')
        plt.plot(np.log10(np.exp(lnT)), planck0, c = 'b', linestyle = '--', 
                 label =  rf'Planck $ \rho $ {np.exp(rho0):.1e}')
        
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
        plt.legend(fontsize = 7)
        # plt.legend( bbox_to_anchor=(1.15,-0.15), ncols = 3,
        #            bbox_transform = ax.transAxes )
        plt.grid()
        plt.savefig('Figs/opacities_comparison.jpg')
        plt.show()
        
    if extrapolation_comp:
        lnT = np.loadtxt(loadpath + 'T.txt')
        lnrho = np.loadtxt(loadpath + 'big_lnrho.txt')
        lnk_planck_inter = RegularGridInterpolator( (lnT, lnrho), lnk_planck)

        plancks = []
        for T in lnT:   
            planck = [ np.log(opacity(T, rho, kind = 'planck', ln = True)) for rho in lnrho]
            plancks.append(planck)
        
        fig, ax = plt.subplots( 1,3 , figsize = (12,4), tight_layout = True, 
                               sharey = True, sharex = True)
        oldrho = np.loadtxt(loadpath + 'rho.txt')

        for planck in plancks:
            ax[0].plot(np.log10(np.exp(lnrho)), planck, c = 'k')
            
        ax[0].axvline( np.log10(np.exp(oldrho[0])), c = 'r')
        ax[0].grid()
        ax[0].set_title( 'Old Extrapolation')
        ax[0].set_xlabel(r'Density $\log_{10}( \rho )$ [g/cm$^3$]')
        ax[0].set_ylabel(r'Opacity $\log_{10}(\kappa)$ [1/cm$^{-1}$]')
        ax[0].set_ylim(-120, 30)
        #######################################################################
        # HOPE EXTRAPOLATION
        #######################################################################
        lnrho = np.loadtxt(loadpath + 'big_lnrho.txt')
        lnk_planck = np.loadtxt(loadpath + 'planck_expansion.txt')
        lnk_planck_inter = RegularGridInterpolator( (lnT, lnrho), lnk_planck)
        
        plancks = []
        for T in lnT:   
            planck = [ np.log(opacity(T, rho, kind = 'planck', ln = True)) for rho in lnrho]
            plancks.append(planck)
            
        oldrho = np.loadtxt(loadpath + 'rho.txt')

        for planck in plancks:
            ax[1].plot(np.log10(np.exp(lnrho)), planck, c = 'k')           
            
        ax[1].axvline( np.log10(np.exp(oldrho[0])), c = 'r')
        ax[1].grid()
        ax[1].set_title( 'NEW Extrapolation, Every T')
        ax[1].set_xlabel(r'Density $\log_{10}( \rho )$ [g/cm$^3$]')
        ax[1].set_ylabel(r'Opacity $\log_{10}(\kappa)$ [1/cm$^{-1}$]')
        
        #######################################################################
        # BASE PLOT
        #######################################################################
        lnT = np.loadtxt(loadpath + 'T.txt')
        lnrho = np.loadtxt(loadpath + 'rho.txt')
        lnk_planck = np.loadtxt(loadpath + 'planck.txt')
        plancks = []
        for i, T in enumerate(lnT):   
            planck = [ np.log( np.exp( lnk_planck[i,j])) for j, rho in enumerate(lnrho)]
            plancks.append(planck)
            
        for planck in plancks:
            ax[2].plot(np.log10(np.exp(lnrho)), planck, c = 'k')      
            
        ax[2].axvline( np.log10(np.exp(lnrho[0])), c = 'r')
        ax[2].grid()
        ax[2].set_title('Data from Elad')
        ax[2].set_xlabel(r'Density $\log_{10}( \rho )$ [g/cm$^3$]')
        ax[2].set_ylabel(r'Opacity $\log_{10}(\kappa)$ [1/cm$^{-1}$]')

    if test_elad: 
        tables = False
        # Load data
        lnT = np.loadtxt(loadpath + 'T.txt')
        lnrho = np.loadtxt(loadpath + 'rho.txt')
        T = np.exp(lnT)
        rho = np.exp(lnrho)
        diff = np.zeros((len(T),len(rho)))
        logT = np.log10(T)
        logrho = np.log10(rho)

        # From Eladython
        def linearpad(D0,z0):
            factor = 100
            dz = z0[-1] - z0[-2]
            # print(np.shape(D0))
            dD = D0[-1,:] - D0[-2,:]
            
            z = [zi for zi in z0]
            z.append(z[-1] + factor*dz)
            z = np.array(z)
            
            D = [di for di in D0]

            D.append(D[-1][:] + factor*dD)
            return np.array(D), np.array(z)

        def pad_interp(x,y,V):
            Vn, xn = linearpad(V, x)
            Vn, xn = linearpad(np.fliplr(Vn), np.flip(xn))
            Vn = Vn.T
            Vn, yn = linearpad(Vn, y)
            Vn, yn = linearpad(np.fliplr(Vn), np.flip(yn))
            Vn = Vn.T
            return x, y, V
        
        rossland = np.loadtxt(loadpath + 'ross.txt')
        T_cool2, Rho_cool2, rossland2 = pad_interp(lnT, lnrho, rossland)
        sigma_rossland = RegularGridInterpolator( (T_cool2, Rho_cool2), rossland2, 
                                            bounds_error= False, fill_value=0)
        
        if tables:
            T_plot = T
            rho_plot = rho
            logT_plot = logT
            logrho_plot = logrho
        else:
            Tmax = np.exp(17.87)
            Tmin = np.exp(8.666)
            pre = '5/'
            fix = '308'
            T_snapplot = np.load(pre + fix + '/T_' + fix + '.npy') #np.linspace(1e5,6e5, 120)
            T_snapplot = T_snapplot[::200]
            rho_snapplot = np.load(pre + fix + '/Den_' + fix + '.npy') #np.linspace(1e-3, 8e-3, 150)
            rho_snapplot = rho_snapplot[::200]*c.en_den_converter
            T_plot = T_snapplot[(T_snapplot<Tmax) & (rho_snapplot<np.max(rho))]
            rho_plot = rho_snapplot[(T_snapplot<Tmax) & (rho_snapplot<np.max(rho))]
            logT_plot = np.log(T_plot)
            logrho_plot = np.log(rho_plot)

        kappa_lte = np.zeros((len(T_plot),len(rho_plot)))
        log_sigma_ross = np.zeros((len(T_plot),len(rho_plot)))
    
        for i in range(len(T_plot)):
            for j in range(len(rho_plot)):
                # Old (till march 2024) code 
                opacity_lte = opacity(T_plot[i], rho_plot[j], 'rosseland', ln = False)
                kappa_lte[i][j] = np.log10(opacity_lte)

                # Eladython
                sigma_rossland_eval = np.exp(sigma_rossland(np.array([np.log(T_plot[i]), np.log(rho_plot[j])]))) #both d and t are in CGS, thus also sigma rosseland and plank
                log_sigma_ross[i][j] = np.log10(sigma_rossland_eval) 
                #diff[i][j] = kappa_lte[i][j] - log_sigma_ross[i][j]

        fig, axs = plt.subplots(1,2, tight_layout = True)
        # Old
        img0 = axs[0].pcolormesh(logrho_plot, logT_plot, kappa_lte, cmap = 'cet_rainbow', vmin = -6, vmax = 4)
        cbar0 = plt.colorbar(img0)
        axs[0].set_xlabel(r'$\log_{10}\rho [g/cm^3]$', fontsize = 12)
        axs[0].set_ylabel(r'$\log_{10}$T [K]', fontsize = 12)
        # cbar0.set_label(r'$\log_{10}\kappa [1/cm]$', fontsize = 10)
        axs[0].title.set_text('Us before')
        # Eladython
        img1 = axs[1].pcolormesh(logrho_plot, logT_plot, log_sigma_ross, cmap = 'cet_rainbow', vmin = -6, vmax = 4)
        cbar1 = plt.colorbar(img1)
        axs[1].set_xlabel(r'$\log_{10}\rho [g/cm^3]$', fontsize = 12)
        cbar1.set_label(r'$\log_{10}\kappa [1/cm]$', fontsize = 10)
        axs[1].title.set_text('Eladython')

        if tables:
            plt.suptitle(r'Opacity using $\rho$,T from tables')
            plt.savefig('CompareOpacityTables.png')
        else: 
            plt.savefig('CompareOpacityFromSnap.png')
        plt.show()

    