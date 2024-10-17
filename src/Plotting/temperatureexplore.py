#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 14:40:34 2024

@author: konstantinos
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 15:50:40 2024

@author: konstantinos
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import colorcet

import src.Utilities.prelude as c

rstar = 0.47
mstar = 0.5
Mbh = '1e+06'
extra = 'beta1S60n1.5Compton'
extra2 = 'beta1S60n1.5ComptonRes20'
simname = f'R{rstar}M{mstar}BH{Mbh}{extra}' 
Mbh = float(Mbh)

simname2 = f'R{rstar}M{mstar}BH{Mbh}{extra2}' 

pre = 'data/ef82/'
eccT = np.loadtxt(f'{pre}eccT{simname}.txt')
days = np.loadtxt(f'{pre}eccdays{simname}.txt')
# ecc2 = np.loadtxt(f'{pre}ecc{simname2}.txt')
# days2 = np.loadtxt(f'{pre}eccdays{simname2}.txt')

Rt = rstar * (Mbh/mstar)**(1/3) # Msol = 1, Rsol = 1
apocenter = Rt * (Mbh/mstar)**(1/3)

radii_start = np.log10(0.4*Rt)
radii_stop = np.log10(apocenter) # apocenter
radii = np.logspace(radii_start, radii_stop, 1000) / apocenter

# # diff = np.abs(ecc[:len(ecc2)] - ecc2)
# # 19 - 19+len(days2) for HR to SHR
# q1 = 1-ecc[19:19+len(ecc2)]
# q2 = 1-ecc2
# ang_mom_deficit = np.abs((q1-q2)) / q1
#%%
fig, ax = plt.subplots(1,1, figsize = (4,4))

# img1 = ax.pcolormesh(radii, days, ang_mom_deficit, 
#                       norm = col.LogNorm(vmin = 1e-3, vmax = 1),
#                       cmap = 'cet_rainbow4')

img1 = ax.pcolormesh(radii, days, np.log10(eccT),  cmap = 'cet_rainbow4',
                     vmin = 4, vmax = 8)
cb = fig.colorbar(img1)
#cb.set_label(r'Ang. Mom. Deficit $\frac{|(1-e_1) - (1-e_2)|}{ 1-e_1 }$ ', fontsize = 14, labelpad = 5)
cb.set_label('logT [K]', fontsize = 14, labelpad = 5)
plt.xscale('log')
plt.ylim(0.12, 1.5)

plt.axvline(Rt/apocenter, c = 'k')
plt.axhline(0.843, c='hotpink')
plt.text(Rt/apocenter + 0.002, 0.3, '$R_\mathrm{T}$', 
         c = 'k', fontsize = 14)

plt.axvline(0.6 * Rt/apocenter, c = 'grey', ls = ':')
plt.text(0.6 * Rt/apocenter + 0.00001, 0.2, '$R_\mathrm{soft}$', 
         c = 'grey', fontsize = 14)
# Axis labels
fig.text(0.5, -0.01, r'r/R$_a$', ha='center', fontsize = 14)
fig.text(-0.02, 0.5, r' Time / Fallback time $\left[ t/t_{FB} \right]$', va='center', rotation='vertical', fontsize = 14)
ax.tick_params(axis = 'both', which = 'both', direction='in')
#ax.set_title(r'$10^6$ M$_\odot$')
m = int(np.log10(Mbh))
plt.title(f'$ 10^{m} M_\odot$')

#%% Check mean temperature

# Get idx for 0.6, 2
radii = np.logspace(radii_start, radii_stop, 1000)  /Rt
idxin = np.argmin(np.abs(radii - 0.6))
idxout = np.argmin(np.abs(radii - 2))


Mbhs = [10_000, 100_000, '1e+06']
cols = ['k', c.AEK, 'maroon']
fig, ax = plt.subplots(1,1, figsize=(4,4))
for i, Mbh in enumerate(Mbhs):
    simname = f'R{rstar}M{mstar}BH{Mbh}{extra}' 
    eccT = np.loadtxt(f'{pre}eccT{simname}.txt')
    days = np.loadtxt(f'{pre}eccdays{simname}.txt')
    Mbh = float(Mbh)
    
    tdisk = eccT.T[idxin:idxout] # get disk T
    #tdisk = np.log10(tdisk)
    mask = tdisk != np.nan
    tdisk = np.ma.masked_where(tdisk == 0, tdisk)
    tdisk = np.log10(np.mean(tdisk, axis=0)) # average over days
    ax.plot(days, tdisk, c = cols[i], lw = 2.5)
ax.set_xlabel(r' Time / Fallback time $\left[ t/t_{FB} \right]$')
ax.set_ylabel('Mean logT in 0.6-2 Rt')
ax.set_ylim(6,7)

# start1 = (6.15-6)
# end1 = start1 + (6.1 - 2/3)
# ax.axvline(1, ymin =start1, ymax = end1, c = 'b', lw = 2, alpha = 1)

#%% Temperature energy vs budget

# Get idx for 0.6, 2
radii = np.logspace(radii_start, radii_stop, 1000)  /Rt
idxin = np.argmin(np.abs(radii - 0.6))
idxout = np.argmin(np.abs(radii - 2))


Mbhs = [10_000, 100_000, '1e+06']
cols = ['k', c.AEK, 'maroon']
fig, ax = plt.subplots(1,1, figsize=(4,4))
for i, Mbh in enumerate(Mbhs):
    simname = f'R{rstar}M{mstar}BH{Mbh}{extra}' 
    eccT = np.loadtxt(f'{pre}eccT{simname}.txt')
    days = np.loadtxt(f'{pre}eccdays{simname}.txt')
    Mbh = float(Mbh)
    ecirc = c.G * Mbh * c.Msol_to_g/(4*Rt*c.Rsol_to_cm) 
    tdisk = eccT.T[idxin:idxout] * c.kb # get disk T
    #tdisk = np.log10(tdisk)
    mask = tdisk != np.nan
    tdisk = np.ma.masked_where(tdisk == 0, tdisk)
    tdisk = np.mean(tdisk, axis=0) # average over days
    ax.plot(days, np.log10(tdisk/ecirc), c = cols[i], lw = 2.5)
ax.set_xlabel(r' Time / Fallback time $\left[ t/t_{FB} \right]$')
ax.set_ylabel('Mean $k_\mathrm{B} T/E_\mathrm{circ}$ in 0.6-2 $R_\mathrm{T}$')
#ax.set_yscale('log')

x1 = [1] * 50
start1 = -25.2
line1 = np.linspace(start1, start1 + 2/3, 50)
plt.plot(x1, line1, c='grey', ls = '--', alpha = 0.8, lw = 2)

x2 = [1.2] * 50
start2 = -24.25
line2 = np.linspace(start2, start2 + 2/3, 50)
plt.plot(x2, line2, c='grey', ls = '--', alpha = 0.8, lw = 2)

plt.text(x1[0]+0.05, start1+0.4,'2/3', c = 'grey', 
         fontsize = 12, fontweight = 3)
plt.text(x2[0]+0.05, start2+0.4,'2/3', c = 'grey', 
         fontsize = 12, fontweight = 3)
    



