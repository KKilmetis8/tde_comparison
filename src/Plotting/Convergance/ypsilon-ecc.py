#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 20:57:12 2023

@author: konstantinos
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 18:38:23 2023

@author: konstantinos
"""

import sys
sys.path.append('/Users/paolamartire/tde_comparison')
import numpy as np
import matplotlib.pyplot as plt
import colorcet
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.figsize'] = [10, 8]
plt.rcParams['axes.facecolor']= 	'whitesmoke'
AEK = '#F1C410'
m = 4
Mbh = 10**m 
Rt =  Mbh**(1/3) # Msol = 1, Rsol = 1
apocenter = 2 * Rt * Mbh**(1/3)  # There is m_* hereeee


# Data Load
radii = np.logspace(np.log10(0.4*Rt), np.log10(apocenter), num=100)
data4 = np.loadtxt('data/ef8/ecc4-fid.txt')
data4CHR = np.loadtxt('data/ef8/ecc4-S60ComptonHires.txt')

# Ranges
fixes4 = np.arange(197, 322+1)
fixes4CHR = np.arange(210, 278+1)
late4 = np.arange(265, 276)
late4CHR = np.arange(268, 278)
mid4 = np.arange(226, 237)
mid4CHR = np.arange(230, 240)
early4 = np.arange(208, 218)
early4CHR = np.arange(210, 220)

# Make Masks
def masker_and_stacker(specific, fixes, data):
    ecc = np.zeros(100)
    for i, snapshot in enumerate(fixes):
        if snapshot in specific:
            idx = np.where(fixes == snapshot)[0][0]
            ecc = np.add(ecc, data[idx])
    inv_len = 1/len(specific)        
    mean_ecc = np.multiply(ecc, inv_len) 
    return mean_ecc


# Early
ecc_early_4 = masker_and_stacker(early4, fixes4, data4)
ecc_early_4CHR = masker_and_stacker(early4CHR, fixes4CHR, data4CHR)
ypsilon_early = 1 - np.divide(ecc_early_4, ecc_early_4CHR)
# Mid
ecc_mid_4 = masker_and_stacker(mid4, fixes4, data4)
ecc_mid_4CHR = masker_and_stacker(mid4CHR, fixes4CHR, data4CHR)
ypsilon_mid = 1 - np.divide(ecc_mid_4, ecc_mid_4CHR)
# Late
ecc_late_4 = masker_and_stacker(late4, fixes4, data4)
ecc_late_4CHR = masker_and_stacker(late4CHR, fixes4CHR, data4CHR)
ypsilon_late = 1 - np.divide(ecc_late_4, ecc_late_4CHR)

#%%
fig, axs = plt.subplots(3,2, tight_layout = True, sharex = True)

# Images
axs[0, 0].plot(radii/apocenter, ypsilon_early, 
        '-', color = 'maroon', 
        markersize = 5)
axs[0, 1].plot(radii/apocenter, ecc_early_4, 
        '-', color = AEK, label = 'Fid',
        markersize = 5)
axs[0, 1].plot(radii/apocenter, ecc_early_4CHR, 
        '--', color = 'k', label = 'CHR')

axs[1, 0].plot(radii/apocenter, ypsilon_mid, 
        '-', color = 'maroon', 
        markersize = 5)
axs[1, 1].plot(radii/apocenter, ecc_mid_4, 
        '-', color = AEK, label = 'Fid',
        markersize = 5)
axs[1, 1].plot(radii/apocenter, ecc_mid_4CHR, 
        '--', color = 'k', label = 'CHR')

axs[2, 0].plot(radii/apocenter, ypsilon_late, 
        '-', color = 'maroon', 
        markersize = 5)
axs[2, 1].plot(radii/apocenter, ecc_late_4, 
        '-', color = AEK, label = 'Fid',
        markersize = 5)
axs[2, 1].plot(radii/apocenter, ecc_late_4CHR, 
        '--', color = 'k', label = 'CHR')

# Title
axs[0,0].set_title('0.78 - 0.87 t/t$_{FB}$', fontsize = 20)
axs[0,1].set_title('0.78 - 0.87 t/t$_{FB}$', fontsize = 20)
axs[1,0].set_title('0.95 - 1.05 t/t$_{FB}$', fontsize = 20)
axs[1,1].set_title('0.95 - 1.05 t/t$_{FB}$', fontsize = 20)
axs[2,0].set_title('1.31 - 1.41 t/t$_{FB}$', fontsize = 20)
axs[2,1].set_title('1.31 - 1.41 t/t$_{FB}$', fontsize = 20)
# Legend
axs[0,1].legend()
axs[1,1].legend()
axs[2,1].legend()

# Grid
axs[0,0].grid()
axs[0,1].grid()
axs[1,0].grid()
axs[1,1].grid()
axs[2,0].grid()
axs[2,1].grid()

# Labels
axs[1,0].set_ylabel(r'$\upsilon$ 1-$e_{fid}$ / 1-$e_{CHR}$',fontsize = 17)
axs[1,1].set_ylabel('1-e', fontsize = 15)
axs[2,1].set_xlabel(r'Radius [$r / R_a$]', fontsize = 17)
axs[2,0].set_xlabel(r'Radius [$r / R_a$]', fontsize = 17)
axs[2,1].set_xscale('log')
