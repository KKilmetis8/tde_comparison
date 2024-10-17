#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 11:50:38 2024

@author: konstantinos
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import colorcet

import src.Utilities.prelude as c
rstar = 0.47
mstar = 0.5
Mbh = 10000
extra = 'beta1S60n1.5Compton'
simname4 = f'data/red/R{rstar}M{mstar}BH{Mbh}{extra}_eladred'
limits4 = [111, 135, 
           136, 156, 
           157, 177, 
           178, 200, 
           201, 221, 
           222, 242,
           243, 263,
           264, 284, 
           285, 305,
           306, 316,
           317, 327, 
           328, 338,
           339, 348, ]
extra = 'beta1S60n1.5ComptonHiRes'

simname4h = f'data/red/R{rstar}M{mstar}BH{Mbh}{extra}_eladred'
limits4h = [80, 90, 
            91, 99,
            100, 110,
            133, 143,
            144, 154,
            155, 165,
            166, 176,
            177, 187,
            188, 198,
            199, 204,
            205, 210]

Mbh = 100_000
extra = 'beta1S60n1.5Compton'
simname5 = f'data/red/R{rstar}M{mstar}BH{Mbh}{extra}_eladred'
limits5 = [130, 150, 
           151, 171,
           172, 190,
           191, 251,
           252, 272,
           273, 278,
           279, 284,
           285, 290,
           291, 295,
           296, 302,
           302, 307,
           308, 314,
           315, 335,
           336, 346,
           347, 357,
           358, 365,]
Mbh = '1e+06'
extra = 'beta1S60n1.5Compton'
simname6 = f'data/red/R{rstar}M{mstar}BH{Mbh}{extra}_eladred'
limits6 = [#180, 194,
           #195, 215,
           216, 218,
           219, 221,
           222, 225,
           226, 229, 
           230, 236,
           237, 257,
           258, 278,
           279, 281,
           282, 289,
           290, 295,
           296, 299,
           300, 310,
           311, 321,
           322, 332,
           333, 343, 
           344, 354, 
           355, 365,
           366, 371,
           372, 379,
           380, 387,
           388, 393,
           394, 398,
           399, 403]

def loader(simname, limits):
    print(simname)
    red = np.loadtxt(f'{simname}_{limits[0]}to{limits[1]}.txt')
    days = np.loadtxt(f'{simname}days_{limits[0]}to{limits[1]}.txt')
    for i in range(2, len(limits), 2):
        try:
            temp_red = np.loadtxt(f'{simname}_{limits[i]}to{limits[i+1]}.txt')
            temp_days = np.loadtxt(f'{simname}days_{limits[i]}to{limits[i+1]}.txt')
        except:
            print('dont have, ', limits[i], limits[i+1])
            continue
        red = np.hstack((red, temp_red))
        days = np.hstack((days, temp_days))
    return red, days

def peak_finder(red, t, lim = 0.77):
    start = np.argmin( np.abs(t - lim) )
    print(start)
    red_g = np.nan_to_num(red[start:])
    t_g = t[start:]
    
    peak = np.argmax(red_g)
    return red[start + peak], t[start + peak]

def Leddington(M):
    return 1.26e38 * M
red4, t4 = loader(simname4, limits4)
peak4, peaktime4 = peak_finder(red4, t4)
red4h, t4h = loader(simname4h, limits4h)
red5, t5 = loader(simname5, limits5)
peak5, peaktime5 = peak_finder(red5, t5)
red6, t6 = loader(simname6, limits6)
peak6, peaktime6 = peak_finder(red6, t6)


# Ours ------------------------------------------------------------------------
# Mbh = 10000
# extra = 'beta1S60n1.5Compton'
# simname4 = f'data/red/redR{rstar}M{mstar}BH{Mbh}{extra}' 
# red4_us = np.loadtxt(f'{simname4}_lums.txt')
# t4_us = np.loadtxt(f'{simname4}_days.txt')

# extra = 'beta1S60n1.5ComptonHiRes'
# simname4h = f'data/red/redR{rstar}M{mstar}BH{Mbh}{extra}' 
# red4h_us = np.loadtxt(f'{simname4h}_lums.txt')
# t4h_us = np.loadtxt(f'{simname4h}_days.txt')

# Mbh = 100_000
# extra = 'beta1S60n1.5Compton'
# simname5 = f'data/red/redR{rstar}M{mstar}BH{Mbh}{extra}' 
# red5h_us = np.loadtxt(f'{simname5}_lums.txt')
# t5h_us = np.loadtxt(f'{simname5}_days.txt')

#%%
fig, ax = plt.subplots(1,1, figsize = (6,4))
ax.plot(t4, red4, c = 'k', marker = '', markersize = 3,
        ls = '-', lw = 4)
ax.plot(peaktime4, peak4, c = 'white', marker = 'X', markersize = 15, 
        markeredgecolor = 'k', markeredgewidth = 2)
ax.axhline(Leddington(1e4), ls = '--', c = 'k')
# ax.plot(t4h, red4h, c = 'forestgreen', marker = 'o', markersize = 3, 
#        ls = '', )
ax.plot(t5, red5, c = c.AEK, marker = 'o', markersize = 3, 
        ls = '', lw = 4)
ax.plot(peaktime5, peak5, c = 'white', marker = 'X', markersize = 15, 
        markeredgecolor = c.AEK, markeredgewidth = 2)
ax.axhline(Leddington(1e5), ls = '--', c = c.AEK)

ax.plot(t6, red6, c = 'maroon', marker = 'o', markersize = 3, 
        ls = '', lw = 4)
ax.plot(peaktime6, peak6, c = 'white', marker = 'X', markersize = 15, 
        markeredgecolor = 'maroon', markeredgewidth = 2)
ax.axhline(Leddington(1e6), ls = '--', c = 'maroon')


# ax.plot(t4_us, red4_us, c = 'k', ls = '--', label = r'$10^4 M_\odot$ us')
# # ax.plot(t4h_us, red4h_us, c = 'forestgreen', ls = '--', label = r'$10^4 M_\odot$ | HR us')
# ax.plot(t5h_us, red5h_us, c = 'b', ls = '-', lw = 3, label = r'$10^5 M_\odot us$')

lw = 5
ax.plot([], [], c = 'k', ls = '-', lw = lw, label = r'$10^4 M_\odot$',)
#ax.plot([], [], c = 'forestgreen', ls = '-', lw = lw, label = r'$10^4 M_\odot$ | HR',)
ax.plot([], [], c = c.AEK, ls = '-', lw = lw, label = r'$10^5 M_\odot$',)
ax.plot([], [], c = 'maroon', ls = '-', lw = lw, label = r'$10^6 M_\odot$',)

#ax.text(1.6, 1e44, r'$10^4 M_\odot$', c = 'k', fontsize = 15)

plt.yscale('log')
plt.ylim(5e41, 3e44)
plt.xlim(0.77,1.75)
plt.title(f'FLD Luminosity')
plt.xlabel('Time [$t_\mathrm{FB}$]', fontsize = 14)
plt.ylabel('Luminosity [erg/s]', fontsize = 14)

#%%
# Lp_by_Ledd = [peak4/Leddington(1e4), peak5/Leddington(1e5), peak6/Leddington(1e6)]
# masses = [1e4, 1e5, 1e6]    

# fig, ax = plt.subplots(1,1, figsize = (3,3))
# ax.scatter(masses[0], Lp_by_Ledd[0], c = 'k', s = 55,)
# ax.scatter(masses[1], Lp_by_Ledd[1], c = c.AEK, s = 55, ec = 'k')
# ax.scatter(masses[2], Lp_by_Ledd[2], c = 'maroon', s = 55, ec = 'k')

# ax.set_ylim(1.1, 1.45)
# ax.set_xscale('log')
# ax.set_ylabel(r'$L_\mathrm{peak}$/$L_\mathrm{Edd}$', fontsize = 15)
# ax.set_xlabel('Black Hole Mass [$M_\odot$]', fontsize = 15)


#plt.legend(bbox_to_anchor=(0.85, 0.75), fontsize = 12, ncols = 1)