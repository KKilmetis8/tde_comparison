"""
Calculate emissivity for a cell

Authors: Paola, Konstantinos
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from src.Optical_Depth.opacity_table import opacity

alpha = 7.5646 * 10**(-15) # radiation density [erg/cm^3K^4]
c = 2.9979e10 # [cm/s] 

def emissivity(T,rho, cell_vol):
    """ Arguments in CGS """
    ln_T = np.log(T)
    ln_rho = np.log(rho)
    ln_planck = opacity(ln_T, ln_rho, 'plank', ln = True)
    k_planck = np.exp(ln_planck) 
    emiss = alpha * c * T**4 * k_planck * cell_vol
    return emiss

if __name__ == "__main__":
    test = emissivity(10**(7),10**(-10),2)
    print(test)