#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 13:16:01 2025

@author: konstantinos
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo  # Using Planck 2018 cosmology
from astropy import units as u

from tqdm import tqdm
import src.Utilities.prelude as c

#%% Get g band luminosity
def g_peakfinder(simname, Mbh, z, plot = False):
    # ZTFg
    
    low_ZTFg = 5.66e14 / (1+z)
    high_ZTFg = 7.48e14 / (1+z)
    less_thanZTFg = c.freqs > low_ZTFg
    more_thanZTFg = c.freqs < high_ZTFg
    ZTFg = less_thanZTFg * more_thanZTFg 
    
    if Mbh == 4:
        fixes = np.arange(80,344+1)
    if Mbh == 5:
        fixes = np.arange(132, 361+1)
    if Mbh == 6:
        fixes = np.arange(180, 444+1)
        
    g_band_max = 0
    for fix in fixes:        
        spectra = np.loadtxt(f'{simname}spectra{fix}.txt')
        for obs in range(c.NPIX): 
            spectrum_of_note = c.freqs * spectra[obs]
            g_luminosity = np.max(spectrum_of_note[ZTFg])
            if g_luminosity > g_band_max:
                g_band_max = g_luminosity
                obs_max = obs
                fix_max = fix
    print(f'Max g-band at snapshot {fix_max} and observer {obs_max}')
    if plot:
        spectra = np.loadtxt(f'{simname}spectra{fix_max}.txt')[obs_max]
        plt.loglog(c.freqs, c.freqs * spectra, c = 'k', label = f'Snapshot:{fix_max} \n Observer:{obs_max}')
        plt.axvspan(5.66e14, 7.48e14, alpha=0.5, color=c.prasinaki, label = 'ZTF g-band')
        plt.ylim(1e39, 1e42)
        plt.xlim(1e14, 1e16)
        plt.title(f'Peak g-band spectrum for $10^{Mbh} \mathrm{{M}}_\odot$')
        plt.ylabel(r'$\nu L_\nu$ [erg/s]')
        plt.xlabel('Frequency [Hz]')
        # plt.text(3e14, 2e39, 'ZTF g-band')
        plt.legend(frameon = False, fontsize = 9)
    return g_band_max
# Mbh = 4
# pre = f'data/blue2/spectra{Mbh}/sumthomp_{Mbh}'
# g_TDE = g_peakfinder(pre, Mbh, z = 1e-4, plot = True)
# l = int(np.log10(g_TDE))
        #%%
def luminosity_to_magnitude(Mbh, z_range=np.logspace(-4, 0, 20)):
    """Convert bolometric luminosity to apparent magnitude as a function of redshift.
    
    Parameters:
        L_bol (float): Bolometric luminosity in Watts.
        z_range (array-like): Array of redshift values to compute magnitudes.

    Returns:
        z_range (numpy array): Redshift values.
        m_app (numpy array): Apparent magnitude values.
    """
    pre = f'data/blue2/spectra{Mbh}/sumthomp_{Mbh}'

    # Sun's bolometric luminosity in erg/s
    L_sun = 3.828e33 
    M_sun_bol = 4.74  # Sun's bolometric magnitude
    
    m_app = np.zeros(len(z_range))
    for i, z in tqdm(enumerate(z_range)):
        g_TDE = g_peakfinder(pre, Mbh, z, plot = False)
        
        # Compute absolute magnitude from luminosity
        M_bol = M_sun_bol - 2.5 * np.log10(g_TDE / L_sun)
        # Compute luminosity distance in parsecs
        D_L = cosmo.luminosity_distance(z_range).to(u.pc).value  # Convert to parsecs
        # Compute apparent magnitude
        m_app = M_bol + 5 * np.log10(D_L / 10)
        
    return z_range, m_app

# Example usage
Mbh = 4
z_vals, m_vals = luminosity_to_magnitude(Mbh)
#%%
# Plot results
plt.figure(figsize=(4, 4))
plt.plot(z_vals, m_vals, '-o', c = 'k', 
         label=f'$M_\mathrm{{BH}}$ = 10$^{Mbh}$ M$_\odot$, $M_*$ = 0.5 M$_\odot$ TDE')
plt.axhline(18.5, c = c.cyan, ls = '--')
plt.axhline(20.8, color = c.prasinaki, ls = '--')

plt.text(1.5e-4, 18.2, 'BTS limit', c = c.cyan)
plt.text(1.5e-4, 20.4, 'ZTFg alert limit', c = c.prasinaki)

plt.axvspan(0.05, 0.15, color = c.reddish, alpha = 0.3, 
            label = '50 \% $z$ of TDEs French+\'20')
plt.axvline(0.08, color = c.reddish, ls = '--', 
            label = 'Median $z$ of TDE French+\'20')

plt.xscale('log')
plt.xlim(1e-4, 1)
plt.xlabel("Redshift (z)")
plt.ylabel("Apparent Magnitude (m)")
plt.ylim(max(m_vals), min(m_vals))  # Invert y-axis for magnitude scale
plt.title(r"Could an IMBH TDE hide in ZTF?",
          fontsize = 8)
plt.legend(frameon = False, fontsize = 9)
