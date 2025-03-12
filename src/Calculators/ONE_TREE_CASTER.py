#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 13:58:33 2023

@author: konstantinos
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gives ray which are logspaced. Around photosphere they sould be 1.
Created on Tue Oct 10 10:19:34 2023

@author: paola, konstantinos

"""
from src.Utilities.isalice import isalice
alice, plot = isalice()

import sys
sys.path.append('/Users/paolamartire/tde_comparison')
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
AEK = '#F1C410'
#%% Constants & Converter
NSIDE = 4
G = 6.6743e-11 # SI
Msol = 1.98847e30 # kg
Rsol = 6.957e8 # m
t = np.sqrt(Rsol**3 / (Msol*G )) # Follows from G = 1
c = 3e8 * t/Rsol # simulator units. Need these for the PW potential
c_cgs = 3e10 # [cm/s]
Msol_to_g = 1.989e33 # [g]
Rsol_to_cm = 6.957e10 # [cm]
den_converter = Msol_to_g / Rsol_to_cm**3


def BONSAI(radii, R, fruit, weight = None):
    """ Outputs are in in solar units """
    gridded_fruit =  np.zeros(len(radii))
    gridded_mass =  np.zeros(len(radii))
    R = np.array([R],).T
    tree = KDTree(R)
    
    if type(weight) == type(None):
        lifting = False
    else:
        lifting = True
        
    for i in range(len(radii)):
        last_progress = 1
        _, idx = tree.query(radii[i])
                            
        # Store
        if lifting:
            gridded_fruit[i] = fruit[idx] * weight[idx]
            gridded_mass[i] = weight[idx]
        else:
            gridded_fruit[i] = fruit[idx]
        
    if lifting:
        gridded_fruit = np.divide(gridded_fruit, gridded_mass)
    return gridded_fruit


    