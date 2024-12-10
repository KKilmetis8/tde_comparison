#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 18:55:57 2024

@author: konstantinos

Loads the opacity data and extrapolates
Quantities X remain in lnX form.
"""

import numpy as np
from src.Opacity.linextrapolator import pad_interp, extrapolator_flipper, nouveau_rich

opac_kind = 'LTE'
opac_path = f'src/Opacity/{opac_kind}_data/'
T_cool = np.loadtxt(f'{opac_path}/T.txt')
Rho_cool = np.loadtxt(f'{opac_path}/rho.txt')
plank = np.loadtxt(f'{opac_path}/planck.txt')
rossland = np.loadtxt(f'{opac_path}/ross.txt')

# T_opac_ex, Rho_opac_ex, rossland_ex = pad_interp(T_cool, Rho_cool, rossland.T)
# _, _, plank_ex = pad_interp(T_cool, Rho_cool, plank.T)

# T_opac_ex, Rho_opac_ex, rossland_ex = extrapolator_flipper(T_cool, Rho_cool, 
#                                                            rossland)
# _, _, plank_ex = extrapolator_flipper(T_cool, Rho_cool, 
#                                       plank)

T_opac_ex, Rho_opac_ex, rossland_ex = nouveau_rich(T_cool, Rho_cool, rossland)
_, _, plank_ex = nouveau_rich(T_cool, Rho_cool, plank)
