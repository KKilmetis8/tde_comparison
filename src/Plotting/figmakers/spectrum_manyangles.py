#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 09:13:46 2025

@author: konstantinos
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import colorcet

import src.Utilities.prelude as c

pre = 'data/blue2/'
m = 5
fix = 288
spectra = np.loadtxt(f'{pre}richex_{m}spectra{fix}.txt')
zsweep = [72, 80, 76, 84, 188, 0] # x, -x, y, -y, z, -z
#zsweep = [72, 104, 136, 168, 180, 188]
fig, ax = plt.subplots(1,1, figsize = (6,4))
colors = [c.darkb, c.cyan, c.prasinaki, c.yellow, c.kroki, c.reddish]
labels = [ 0, 18, 36, 54, 72, 90]
zsweep = [104, 136, (152, 167), 
          (168, 179), 
          (180, 187), 
          (188, 191), 140]
for i,z in enumerate((zsweep)):
    try:
        l = len(z)
        print(zsweep[i][0])
        spectrum = 0.5 * ( spectra[ zsweep[i][0] ] + spectra[zsweep[i][1] ])
        print(zsweep[i])
    except:
        spectrum = spectra[ zsweep[i] ] 
    ax.plot(c.freqs, c.freqs * spectrum, 
            lw = 2) #, c = colors[i], label = f'{labels[i]}')



# Make nice
ax.legend(bbox_to_anchor = (1.08,0.1,0.1,0.6), ncols = 1)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(1e40, 4e43)
ax.set_xlim(2e13, 1e21)

ax.set_title(f'Spectrum {m} {fix}')
plt.xlabel('Frequency [Hz]', fontsize = 14)
plt.ylabel('Luminosity [erg/s]', fontsize = 14)

ypos = 9e42
ax.axvspan(4e14, 7e14, alpha=0.2, color='gold')
ax.text(3e14, ypos, 'Vis.', fontsize = 12)

ax.axvspan(7e14, 7e16, alpha=0.2, color='purple')
ax.text(5e15, ypos, 'UV', fontsize = 12)

ax.axvspan(7e16, 5e18, alpha=0.2, color='cyan')
ax.text(3e17, ypos, 'Soft X', fontsize = 12)

ax.axvspan(5e18, 5e19, alpha=0.2, color='b')
ax.text(5.8e18, ypos, 'Hard X', fontsize = 12)
    
#%%
import pandas as pd
def tuple_parse(strings):
    ''' parses "(1,2,3)" '''
    xs = np.zeros(len(strings))
    ys = np.zeros(len(strings))
    zs = np.zeros(len(strings))

    for i, string in enumerate(strings):
        values = string.strip("()").split(", ")
        tuple_values = tuple(np.float64(value.split("(")[-1].strip(")")) for value in values)
        xs[i] = tuple_values[0]
        ys[i] = tuple_values[1]
        zs[i] = tuple_values[1]

    return xs, ys, zs
df5 = pd.read_csv('data/photosphere/richex_photocolor5.csv', sep = ',',
                   comment = '#', header = None)
idx5 = np.argmin(np.abs(fix - df5.iloc[:,0]))
x, y, z = tuple_parse(df5.iloc[idx5][-192:])
colorsphere = np.sqrt(x**2+y**2+z**2)
plt.plot(np.arange(192), np.log10(colorsphere), 'o', c = 'k', markersize = 2, lw = 0.75)
print(colorsphere[188])

