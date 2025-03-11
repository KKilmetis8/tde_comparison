#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 14:07:08 2025

@author: konstantinos
"""

# Vanilla
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import colorcet
import healpy as hp

# Choc
import src.Utilities.prelude as c

# Constants
rstar = 0.47
mstar = 0.5
Rt4 = rstar * (1e4/mstar)**(1/3)
amin4 = 0.5 * Rt4 * (1e4/mstar)**(1/3)
Rt5 = rstar * (1e5/mstar)**(1/3)
amin5 =  0.5 * Rt5 * (1e5/mstar)**(1/3)
Rt6 = rstar * (1e6/mstar)**(1/3)
amin6 =  0.5 * Rt6 * (1e6/mstar)**(1/3)
def find_sph_coord(r, theta, phi):
    x = r * np.sin(np.pi-theta) * np.cos(phi) #Elad has just theta
    y = r * np.sin(np.pi-theta) * np.sin(phi)
    z = r * np.cos(np.pi-theta)
    return [x,y,z]

def tuple_parse(strings):
    ''' parses "(1,2,3)" '''
    xs = np.zeros(len(strings))
    ys = np.zeros(len(strings))
    for i, string in enumerate(strings):
        values = string.strip("()").split(", ")
        tuple_values = tuple(np.float64(value.split("(")[-1].strip(")")) for value in values)
        xs[i] = tuple_values[0]
        ys[i] = tuple_values[1]
    return xs, ys


pre = 'data/denproj/paper/'
suf = 'beta1S60n1.5Compton'
when = 'late' # choices: early mid late test
plane = 'XY'
photo = False
if when == 'early':
    # 0.42, 0.82, 1.22
    f4 = 179# 240, 300] 
    f5 = 227# 288, 349] # 0.5, 1, 1.4
    f6 = 315# 379, 444] # 0.5, should be 376 and 414 but ok
if when == 'mid':
    # 0.42, 0.82, 1.22
    f4 = 240# 240, 300] 
    f5 = 288# 288, 349] # 0.5, 1, 1.4
    f6 = 379 # 0.5, should be 376 and 414 but ok
if when == 'late':
    # 0.42, 0.82, 1.22
    f4 = 300# 240, 300] 
    f5 = 349# 288, 349] # 0.5, 1, 1.4
    f6 = 444# 379, 444] # 0.5, should be 376 and 414 but ok
    title_txt = 'Time: Trial t/t$_{FB}$'

size = 6
step = 1
fontsize = 15
# fig, ax = plt.subplots(figsize = (0.8*size, size),  constrained_layout = True)
fig = plt.figure(figsize = (1.2*size, size), constrained_layout=True, dpi = 400)
import matplotlib.gridspec as gridspec
spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
ax4 = fig.add_subplot(spec[0, 0])
ax6 = fig.add_subplot(spec[1, :])
ax5 = fig.add_subplot(spec[0, 1])

# Load projection data
den4 = np.loadtxt(f'{pre}4normal{f4}.txt')[::step].T[::step].T
x4 = np.loadtxt(f'{pre}4normalx.txt')[::step]
y4 = np.loadtxt(f'{pre}4normaly.txt')[::step]

den5 = np.loadtxt(f'{pre}5normal{f5}.txt')[::step].T[::step].T
x5 = np.loadtxt(f'{pre}5normalx.txt')[::step]
y5 = np.loadtxt(f'{pre}5normaly.txt')[::step]

den6 = np.loadtxt(f'{pre}6normal{f6}.txt')[::step].T[::step].T
x6 = np.loadtxt(f'{pre}6normalx.txt')[::step]
y6 = np.loadtxt(f'{pre}6normaly.txt')[::step]

# norm
norm4 = mstar * c.Msol_to_g/(amin4 * c.Rsol_to_cm)**2
norm5 = mstar* c.Msol_to_g/(amin5 * c.Rsol_to_cm)**2
norm6 = mstar* c.Msol_to_g/(amin6 * c.Rsol_to_cm)**2
den4 = np.log10(10**den4 / norm4)
den5 = np.log10(10**den5 / norm5)
den6 = np.log10(10**den6 / norm6)

reddata4 = np.genfromtxt(f'data/red/red_richex{4}.csv', delimiter = ',').T
tidx4 = np.argmin(np.abs(f4 - reddata4[0]))
time4 = reddata4[1][tidx4]
reddata4 = np.genfromtxt(f'data/red/red_richex{5}.csv', delimiter = ',').T
tidx5 = np.argmin(np.abs(f5 - reddata4[0]))
time5 = reddata4[1][tidx5]
reddata6 = np.genfromtxt(f'data/red/red_richex{6}.csv', delimiter = ',').T
tidx6 = np.argmin(np.abs(f6 - reddata6[0]))
time6 = reddata6[1][tidx6]

# ax4.text(0.78, 0.8, f'{time4:.2f} $t_\mathrm{{FB}}$', 
#              fontsize = 12, c = 'white', transform = ax4.transAxes)
# ax5.text(0.78, 0.8, f'{time4:.2f} $t_\mathrm{{FB}}$',
#              fontsize = 12, c = 'white', transform = ax5.transAxes)
ax6.text(0.1, 0.1, f'{time4:.2f} $t_\mathrm{{FB}}$',
             fontsize = fontsize + 3, c = 'white', transform = ax6.transAxes)

# Plot projection data
dmin = -5
dmax = 0
img = ax4.pcolormesh(x4/amin4, y4/amin4, den4.T, cmap = 'cet_fire',
                       vmin = dmin, vmax = dmax)
ax5.pcolormesh(x5/amin5, y5/amin5, den5.T, cmap = 'cet_fire',
                         vmin = dmin, vmax = dmax)
ax6.pcolormesh(x6/amin6, y6/amin6, den6.T, cmap = 'cet_fire',
                         vmin = dmin, vmax = dmax)

# Plot Rt
ax4.add_patch(mp.Circle((0,0), Rt4/amin4, ls = '-', 
                            color = 'c', fill = False, lw = 1))
ax5.add_patch(mp.Circle((0,0), Rt5/amin5, ls = '-', 
                            color = 'c', fill = False, lw = 1))
ax6.add_patch(mp.Circle((0,0), Rt6/amin6, ls = '-',
                            color = 'c', fill = False, lw = 1))

xmin = -1.1
xmax = 0.3
ymin = -0.4
ymax = 0.4
ax4.set_xlim(xmin, xmax)
ax4.set_ylim(ymin, ymax)
ax5.sharex(ax4)
ax5.sharey(ax4)
ax6.sharex(ax4)
ax6.sharey(ax4)



ax4.text(-0.1, 0.27, '10$^4$ M$_\odot$', fontsize = fontsize + 4, c  = 'white')
ax5.text(-0.1, 0.27, '10$^5$ M$_\odot$', fontsize = fontsize + 4, c  = 'white')
ax6.text(0.05, 0.27, '10$^6$ M$_\odot$', fontsize = fontsize + 4, c  = 'white')

ax6.set_xlabel(r'X $[\alpha_\mathrm{min}]$', fontsize = fontsize)
ax6.set_ylabel(r'Y $[\alpha_\mathrm{min}]$', fontsize = fontsize)
cb = fig.colorbar(img, cax=fig.add_axes([1, 0.08, 0.05, 0.905]))
cb.set_label(r'Normalized surface density, $\log_{10} (\Sigma \frac{\alpha_\mathrm{min}^2}{m_*})$', fontsize = fontsize)
cb.ax.tick_params(labelsize = fontsize-4)
for oneax in [ax4, ax5, ax6]:#ax.flatten():
    oneax.tick_params(top=True, right=True, colors = 'whitesmoke',  
                      labelcolor = 'k', labelsize = fontsize - 4)


    
    
    
