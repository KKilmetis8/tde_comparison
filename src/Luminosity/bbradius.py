#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 17:20:26 2025

@author: konstantinos
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')
from lmfit import Model

import src.Utilities.prelude as c
def Planck(n, T):
    # reyleigh = 2 * n**3 / c.c**2
    # wien =  1/ ( np.exp(c.h * n / (c.kb * T)) -1)
    return  2 * n**3 / c.c**2  * 1/ ( np.exp(c.h * n / (c.kb * T)) -1)
pmodel = Model(Planck)
def R_blackbody(SED, T):
    L = np.trapezoid(SED * c.freqs, c.freqs)
    Rbb = np.sqrt(L / (c.stefan * 4 * np.pi * T**4))
    return Rbb

def telescope_masks(telescope):
    less_thanZTFr = c.freqs > 4.11e14
    more_thanZTFr = c.freqs < 5.07e14
    ZTFr = less_thanZTFr * more_thanZTFr # AND
    
    less_thanZTFg = c.freqs > 5.66e14
    more_thanZTFg = c.freqs < 7.48e14
    ZTFg = less_thanZTFg * more_thanZTFg 
    
    ZTF = ZTFg + ZTFr # OR
    
    less_thanULTRASAT = c.freqs > 1.03e15
    more_thanULTRASAT = c.freqs < 1.3e15
    ULTRASAT = less_thanULTRASAT * more_thanULTRASAT
    
    # T.S. Poole+'07, it's in Angstroms
    less_SWIFTv = c.freqs < c.c / (1e-8 * (5468-749)) 
    more_SWIFTv = c.freqs > c.c / (1e-8 * (5468+749)) 
    SWIFTv = less_SWIFTv * more_SWIFTv
    
    less_SWIFTb = c.freqs < c.c / (1e-8 * (4392-975)) 
    more_SWIFTb = c.freqs > c.c / (1e-8 * (4392+975)) 
    SWIFTb = less_SWIFTb * more_SWIFTb
    
    less_SWIFTu = c.freqs < c.c / (1e-8 * (3465-785)) 
    more_SWIFTu = c.freqs > c.c / (1e-8 * (3465+785)) 
    SWIFTu = less_SWIFTu * more_SWIFTu
    
    less_SWIFTuvw1 = c.freqs < c.c / (1e-8 * (2600-693)) 
    more_SWIFTuvw1 = c.freqs > c.c / (1e-8 * (2600+693)) 
    SWIFTuvw1 = less_SWIFTuvw1 * more_SWIFTuvw1
    
    less_SWIFTuvm2 = c.freqs < c.c / (1e-8 * (2246-498)) 
    more_SWIFTuvm2 = c.freqs > c.c / (1e-8 * (2246+498)) 
    SWIFTuvm2 = less_SWIFTuvm2 * more_SWIFTuvm2
    
    less_SWIFTuvw2 = c.freqs < c.c / (1e-8 * (1928-657)) 
    more_SWIFTuvw2 = c.freqs > c.c / (1e-8 * (1928+657)) 
    SWIFTuvw2 = less_SWIFTuvw2 * more_SWIFTuvw2

    less_thanSWIFTx = c.freqs > 0.3 * 1000 / c.Hz_to_ev
    more_thanSWIFTx = c.freqs < 10 * 1000 / c.Hz_to_ev
    SWIFTx = less_thanSWIFTx * more_thanSWIFTx
    
    SWIFT = SWIFTv + SWIFTb + SWIFTu + SWIFTuvw1 + SWIFTuvm2 + SWIFTuvw2 + SWIFTx
    if telescope == 'ZTF':
        return ZTF
    elif telescope == 'ULTRASAT':
        return ULTRASAT
    elif telescope == 'SWIFT':
        return SWIFT
    elif telescope == 'SWIFTv':
        return SWIFTv
    elif telescope == 'SWIFTb':
        return SWIFTb
    elif telescope == 'SWIFTu':
        return SWIFTu
    elif telescope == 'SWIFTuvw1':
        return SWIFTuvw1
    elif telescope == 'SWIFTuvm2':
        return SWIFTuvm2
    elif telescope == 'SWIFTuvw2':
        return SWIFTuvw2
    elif telescope == 'SWIFTx':
        return SWIFTx
    elif telescope == 'UV':
        return ZTF + ULTRASAT  + SWIFTv  + SWIFTb   + SWIFTu + SWIFTuvw1 + SWIFTuvm2
    elif telescope == 'all':
        return ZTF + ULTRASAT + SWIFT
    else:
        raise ValueError('Telescope not available. Choose from ZTF, ULTRASAT, SWIFT, all.')
pre = 'data/bluepaper/'
f4 = 240
spectra4 = np.loadtxt(f'{pre}local_4spectra{f4}.txt')
mask = telescope_masks('UV')
zsweep = [104] # , 136, 152, 167, 168, 179, 180, 187, 188, 191]#, 140]
Rbbs = []
for z in zsweep:
    # Tfit = curve_fit(Planck, c.freqs[mask], spectra4[z][mask], p0 = [1000], )[0][0]
    fit = pmodel.fit(spectra4[z][mask], n = c.freqs[mask], T = 50_000)
    Tfit = fit.params['T'].value
    BBfit = [ Planck(freq, Tfit) for freq in c.freqs]
    Rbbs.append(R_blackbody(BBfit, Tfit))
print(f'Tfit {Tfit} K')
print(f'Average BB radius {np.mean(Rbbs)/ c.Rsol_to_cm:.2e} RSol')
plt.figure()
plt.plot(c.freqs[mask], spectra4[z][mask], ':h', c = 'k',markersize = 3, label = 'Spectrum' )
plt.plot(c.freqs[mask], fit.best_fit, c='r', ls = '--', label = 'Best Fit')
plt.plot(c.freqs[mask], fit.init_fit, c='b', ls = '--', label = 'Init. Fit')
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e23, 1e27)
plt.legend(frameon=False)
#%% Selector test

fig, ax = plt.subplots(1,1, figsize = (4,4))

ZTFmask = telescope_masks('ZTF')
plt.plot(c.freqs, spectra4[104] , c = 'k', label = 'Spectrum', lw = 1)
plt.plot(c.freqs[ZTFmask], spectra4[104][ZTFmask], lw = 4, 
         c = c.AEK, ls = '-', label = 'ZTF')

ULTRASATmask = telescope_masks('ULTRASAT')
plt.plot(c.freqs[ULTRASATmask], spectra4[104][ULTRASATmask], 
         c = 'purple', ls = '-', label = 'ULTRASAT', lw = 4)



SWIFTvmask = telescope_masks('SWIFTv')
plt.plot(c.freqs[SWIFTvmask], spectra4[104][SWIFTvmask],
         c = 'green', ls = '-', label = 'SWIFTv', lw = 2)

SWIFTbmask = telescope_masks('SWIFTb')
plt.plot(c.freqs[SWIFTbmask], spectra4[104][SWIFTbmask],
         c = 'cornflowerblue', ls = '-', label = 'SWIFTb', lw = 2)

SWIFTumask = telescope_masks('SWIFTu')
plt.plot(c.freqs[SWIFTumask], spectra4[104][SWIFTumask],
         c = 'mediumpurple', ls = '-', label = 'SWIFTu', lw = 2)

SWIFTuvw1mask = telescope_masks('SWIFTuvw1')
plt.plot(c.freqs[SWIFTuvw1mask], spectra4[104][SWIFTuvw1mask],
         c = 'magenta', ls = '-', label = 'SWIFTuvw1', lw = 2)

SWIFTuvm2mask = telescope_masks('SWIFTuvm2')
plt.plot(c.freqs[SWIFTuvm2mask], spectra4[104][SWIFTuvm2mask],
         c = 'indianred', ls = '-', label = 'SWIFTuvm2', lw = 2)

SWIFTuvw2mask = telescope_masks('SWIFTuvw2')
plt.plot(c.freqs[SWIFTuvw2mask], spectra4[104][SWIFTuvw2mask],
         c = 'grey', ls = '-', label = 'SWIFTuvw2', lw = 2)

SWIFTmask = telescope_masks('SWIFTx')
plt.plot(c.freqs[SWIFTmask], spectra4[104][SWIFTmask],
         c = 'skyblue', ls = '-', label = 'SWIFTx', lw = 2)

Tfit = curve_fit(Planck, c.freqs[mask], spectra4[104][mask], p0 = [4000])[0][0]
BBfit = [ Planck(freq, Tfit) for freq in c.freqs]
plt.plot(c.freqs, BBfit, c = 'r', ls = '--', label = 'BB fit')







plt.legend(frameon = False, ncols = 1, fontsize = 7)
plt.loglog()
lowx = 2e13
highx = 5e19
plt.xlim(lowx, highx)
plt.ylim(1e5, 1e27)
plt.xlabel('Frequency [Hz]')
plt.ylabel(r'$L_\nu$ [erg/s Hz]')

axins = ax.inset_axes([0.6, 0.6, 0.4, 0.4],)


ZTFmask = telescope_masks('ZTF')
axins.plot(c.freqs, spectra4[104] , c = 'k', label = 'Spectrum', lw = 1)
axins.plot(c.freqs[ZTFmask], spectra4[104][ZTFmask], lw = 4, 
         c = c.AEK, ls = '-', label = 'ZTF')

ULTRASATmask = telescope_masks('ULTRASAT')
axins.plot(c.freqs[ULTRASATmask], spectra4[104][ULTRASATmask], 
         c = 'purple', ls = '-', label = 'ULTRASAT', lw = 4)
SWIFTvmask = telescope_masks('SWIFTv')
axins.plot(c.freqs[SWIFTvmask], spectra4[104][SWIFTvmask],
         c = 'green', ls = '-', label = 'SWIFTv', lw = 2)

SWIFTbmask = telescope_masks('SWIFTb')
axins.plot(c.freqs[SWIFTbmask], spectra4[104][SWIFTbmask],
         c = 'cornflowerblue', ls = '-', label = 'SWIFTb', lw = 2)

SWIFTumask = telescope_masks('SWIFTu')
axins.plot(c.freqs[SWIFTumask], spectra4[104][SWIFTumask],
         c = 'mediumpurple', ls = '-', label = 'SWIFTu', lw = 2)

SWIFTuvw1mask = telescope_masks('SWIFTuvw1')
axins.plot(c.freqs[SWIFTuvw1mask], spectra4[104][SWIFTuvw1mask],
         c = 'magenta', ls = '-', label = 'SWIFTuvw1', lw = 2)

SWIFTuvm2mask = telescope_masks('SWIFTuvm2')
axins.plot(c.freqs[SWIFTuvm2mask], spectra4[104][SWIFTuvm2mask],
         c = 'indianred', ls = '-', label = 'SWIFTuvm2', lw = 2)

SWIFTuvw2mask = telescope_masks('SWIFTuvw2')
axins.plot(c.freqs[SWIFTuvw2mask], spectra4[104][SWIFTuvw2mask],
         c = 'grey', ls = '-', label = 'SWIFTuvw2', lw = 2)

axins.plot(c.freqs, BBfit, c = 'r', ls = '--', label = 'BB fit')
ax.set_title('State of the Fit')
lowx = 2e14
highx = 3e15
axins.set_xlim(lowx, highx)
axins.set_ylim(1e24, 3e25)
axins.loglog()
ax.indicate_inset_zoom(axins, edgecolor="black")


