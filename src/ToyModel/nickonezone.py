#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 12:32:15 2024

@author: konstantinos
"""

# Vanilla
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Choc
import src.Utilities.prelude as c

# Functions 
def af(t, Mbh):
    return (t/2*np.pi)**(2/3) * (c.Gcgs * Mbh)**(1/3)

def Leddington(Mbh):
    return 1.26e38 * (Mbh /c.Msol_to_g)

def tfb(Mbh, mstar = 0.5, rstar = 0.47):
    tfb = 40 * np.power( Mbh/1e6, 1/2) * np.power(mstar,-1) \
        * np.power(rstar, 3/2)
    tfb *= 24*60*60
    return tfb

def Mf(t, mstar, tfb_this):
    term1 = mstar/(3*tfb_this)
    term2 = (t/tfb_this)**(-5/3)
    return term1*term2
    
def onezone(t, y, Mbh, mstar, rstar, tfb_this):
    ac, Mc = y
    if Mc < mstar/2:
        Mf_this = Mf(t, mstar, tfb_this)
    else:
        Mf_this = 0
    term1 = - 2*ac**2*Leddington(Mbh)/(c.Gcgs * Mbh * Mc)
    term2 = Mf_this/Mc * ac * (1 - ac/af(t, Mbh))
    return [term1 + term2, Mf_this]

# Config
mstar = 0.5 * c.Msol_to_g
rstar = 0.47 * c.Rsol_to_cm
fig, axs = plt.subplots(1,3, figsize = (10,4), tight_layout = True, sharex=True,
                        sharey=True)

# Solve
for Mbh, ax in zip([1e4,1e5,1e6], axs):
    Mbh *= c.Msol_to_g
    Rt = rstar * (Mbh/mstar)**(1/3)
    amin = 0.5 * Rt * (Mbh/mstar)**(1/3)
    
    # tfb
    Mbh_n = Mbh / c.Msol_to_g
    mstar_n = mstar / c.Msol_to_g
    rstar_n = rstar / c.Rsol_to_cm
    tfb_this = tfb(Mbh_n, mstar_n, rstar_n)
    
    sol = solve_ivp(onezone, 
                    t_span = [1*tfb_this, 2e3*tfb_this], 
                    y0 = [amin, mstar*1e-3], # Init. ac = amin, Mc = 0
                    first_step = tfb_this*1e-4, max_step = tfb_this*1e-1,
                    args = (Mbh, mstar, rstar, tfb_this),
                    dense_output=True)

    # Plot
    Rc = 2*Rt / Rt
    time = sol.t / tfb_this
    ac = sol.y[0] / Rt
    Mc = sol.y[1] / mstar
    
    ax.plot(time, ac, c='k', lw = 2)
    ax.axhline(Rc, c = 'r', ls ='--')
    ax2 = ax.twinx()
    ax2.plot(time, Mc, c=c.AEK, lw = 2)
    
    idx = np.argmin(np.abs(ac - Rc))
    ax.plot(time[idx], ac[idx], 'h', c='r', markeredgecolor='k', 
            markersize = 10, markeredgewidth = 2)
    ax.text(time[idx], ac[idx]+5, 
            f'$t_\mathrm{{circ}}$ = {time[idx]:.0f} $t_\mathrm{{FB}}$',
            c = 'k', fontsize = 10)
    # Text
    ax.set_xscale('log')
    ax.set_xlabel(r'Time [$t_\mathrm{FB}]$')
    ax.set_ylabel(r'$a_c$ [$R_\mathrm{T}$]')
    ax2.set_ylabel(r'Cloud Mass [$M_*$]')
    m = int(np.log10(Mbh/c.Msol_to_g))
    ax.set_title(f'$M_\mathrm{{BH}}$ = $10^{m} M_\odot$')
































