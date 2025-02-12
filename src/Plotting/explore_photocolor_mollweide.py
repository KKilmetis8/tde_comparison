#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 14:59:06 2025

@author: konstantinos
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:11:11 2025

@author: konstantinos
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.colors as col
import colorcet
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import src.Utilities.prelude as c

pre = 'data/bluepaper/'
thetas = np.zeros(c.NPIX)
phis = np.zeros(c.NPIX) 
for iobs in range(0,c.NPIX):
    theta, phi = hp.pix2ang(c.NSIDE, iobs) # theta in [0,pi], phi in [0,2pi]
    thetas[iobs] = theta
    phis[iobs] = phi
    
m = 4
mstar = 0.5
rstar = 0.47

Rt = rstar * (10**m/mstar)**(1/3)
amin = Rt * (10**5/mstar)**(1/3)
if m == 4:
    fixes4 = [179, 240, 300] 
    fixes  = fixes4
if m == 5:
    fixes5 = [227, 288, 349]
    fixes = fixes5
if m == 6:
    fixes6 = [315, 379, 444]
    fixes = fixes6
reddata4 = np.genfromtxt(f'data/red/red_richex{4}.csv', delimiter = ',').T


fig, axs = plt.subplots(nrows = 3, ncols=2, figsize = (8,8))    

for f, i in zip(fixes, range(len(fixes))):
    color4 = pd.read_csv(f'{pre}localbig_{m}photocolor{f}.csv', header = None).iloc[-1,6]
    photo4 = pd.read_csv(f'{pre}localbig_{m}photocolor{f}.csv', header = None).iloc[-1,5]
    color4 = np.array(list(map(float, color4.strip("[] \n").split()))) / amin
    photo4 = np.array(list(map(float, photo4.strip("[] \n").split()))) / amin

    # Time text
    tidx4 = np.argmin(np.abs(f - reddata4[0]))
    time4 = reddata4[1][tidx4]
    
    plt.axes(axs[i,0])
    hp.mollview(photo4, fig=fig.number, hold=True, cbar = True, 
                cmap = 'cet_CET_L10', min = 0, max = 5,  unit = r'$\alpha_\mathrm{min}$',
                title = f'Photosphere $10^{m} M_\odot$  {time4:.2f} $t_\mathrm{{FB}}$')
    hp.graticule()
    plt.axes(axs[i,1])
    hp.mollview(color4, fig=fig.number, hold=True, cbar = True, 
                cmap = 'cet_CET_L10', min = 0, max = 5,  unit = r'$\alpha_\mathrm{min}$',
                title = f'Colorsphere $10^{m} M_\odot$  {time4:.2f} $t_\mathrm{{FB}}$')
    hp.graticule()
    

# ax[2,1].set_xlabel(r'X [x/$\alpha_\mathrm{min}$]', fontsize = 17)
# ax[1,0].set_ylabel(r'Z [z/$\alpha_\mathrm{min}$]', fontsize = 17)
# ax[0,0].set_title('$10^4 M_\odot$', fontsize = 17)
# ax[0,1].set_title('$10^5 M_\odot$', fontsize = 17)
# ax[0,2].set_title('$10^6 M_\odot$', fontsize = 17)
# ax[0,1].set_xlim(-0.05, 1)
# ax[0,1].set_ylim(-0.05, 0.4)
#%%

fig, axs = plt.subplots(nrows = 3, figsize = (8,8))    
for f, i in zip(fixes, range(len(fixes))):
    color4 = pd.read_csv(f'{pre}localbig_{m}photocolor{f}.csv', header = None).iloc[-1,6] 
    photo4 = pd.read_csv(f'{pre}localbig_{m}photocolor{f}.csv', header = None).iloc[-1,5]
    # print(photo4)
    color4 = np.array(list(map(float, color4.strip("[] \n").split()))) 
    photo4 = np.array(list(map(float, photo4.strip("[] \n").split()))) 
    ratio = photo4/color4

    # Time text
    tidx4 = np.argmin(np.abs(f - reddata4[0]))
    time4 = reddata4[1][tidx4]
    
    plt.axes(axs[i])
    hp.mollview(ratio, fig=fig.number, hold=True, cbar = True, min = 0, max = 10,
                cmap = 'cet_CET_L10', 
                title = f'Photo/Color $10^{m} M_\odot$  {time4:.2f} $t_\mathrm{{FB}}$')
    hp.graticule()

    
    
    
    