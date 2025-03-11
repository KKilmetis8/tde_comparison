#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 16:02:30 2025

@author: konstantinos
"""

import numpy as np
import matplotlib.pyplot as plt
import src.Utilities.prelude as c 

plt.figure(figsize = (3,3), dpi = 500)

smbh = np.logspace(6, 10)
imbh = np.logspace(3.5, 6)
fontsize = 12

# plt.fill_between(smbh, 1/smbh,  2/smbh, color = 'k', alpha = 0.5)

plt.axvspan(5e3,1e6, color = 'k', alpha = 0.15)
plt.text(1e5, 2.5e-10, 'IMBHs \n (unmeasured)', c = 'k', fontsize = fontsize -2, 
         horizontalalignment = 'center')
plt.text(1e7, 7e-10, 'SMBHs', c = 'k', fontsize = fontsize -2, 
         horizontalalignment = 'center')
plt.loglog()
plt.xlim(6e3, 1e8)
plt.ylim(1e-10, 1e-3)
plt.xlabel('Black Hole Mass  [M$_\odot$]' , fontsize = fontsize)
plt.ylabel('Number of Black Holes', fontsize = fontsize)


# plt.fill_between(imbh, 1/imbh, 2/imbh, color = 'b', alpha = 0.2)
# plt.text(1e4, 6e-5, 'Pop. III', color = 'b', fontsize = fontsize - 4,
#          rotation = -32)
# plt.fill_between(imbh, 1e-6, 2e-6, color = 'g', alpha = 0.2)
# plt.text(1e4, 1.15e-6, 'Grav. Runaway', color = 'g', fontsize = fontsize - 5,
#          rotation = 0)
# plt.fill_between(imbh, 1e-12*imbh, 1e-12*2*imbh, color = 'r', alpha = 0.2)
# plt.text(1e4, 4e-9, 'Dir. Collapse', color = 'r', fontsize = fontsize - 4,
#          rotation = 32)

plt.tick_params( axis='y', which='both', left=False, labelleft=False) 