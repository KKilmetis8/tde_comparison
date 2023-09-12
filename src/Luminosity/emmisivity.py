"""
Calculate emissivity for a cell
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

alpha = 7.5646 * 10**(-15) #radiation density [erg/cm^3K^4]
solar_mass = 1.989e33 #[g]
solar_radius = 6.957e10 #[cm]
G = 6.6743e-11 #SI
c = 2.9979e10 #[cm/s] 
t_conv = np.sqrt((solar_radius/10**2)**3/(G*solar_mass*10**(-3))) #time in seconds is time_simlation*t_conv

"""Import data from simulation. NB: Planck, T, rho are in ln"""
loadpath = 'src/Optical_Depth/'
lnT = np.loadtxt(loadpath + 'T.txt') 
lnrho = np.loadtxt(loadpath + 'rho.txt')
lnplanck = np.loadtxt(loadpath + 'planck.txt')

lnk_inter = RegularGridInterpolator( (lnT, lnrho), lnplanck) #we use T and rho to interpolate??

def emissivity(T,rho, cell_vol):
    """this function accepts arguments in CGS"""
    ln_T = np.log(T)
    ln_rho = np.log(rho)
    ln_planck = lnk_inter((ln_T, ln_rho))
    k_planck = np.exp(ln_planck) 
    emiss = alpha * c * T**4 * k_planck * cell_vol
    return emiss

if __name__ == "__main__":
    test = emissivity(10**(7),10**(-10),2)
    print(test)