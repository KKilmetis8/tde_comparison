""" Interpolate and extrapolate opacity for low density"""
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

loadpath = 'src/Optical_Depth/'
lnrho = np.loadtxt(loadpath + 'rho.txt')
lnk_ross = np.loadtxt(loadpath + 'ross.txt')[0]
lnk_planck = np.loadtxt(loadpath + 'planck.txt')
lnk_scatter = np.loadtxt(loadpath + 'scatter.txt')

natural_ross = CubicSpline(lnrho,lnk_ross, bc_type='natural')
natural_planck = CubicSpline(lnrho,lnk_planck, bc_type='natural')
natural_scatter = CubicSpline(lnrho,lnk_scatter, bc_type='natural')

def opacity(rho, kind, ln = True) -> float:
    '''
    Return the rosseland mean opacity in [cgs], given a value of density,
    temperature and and a kind of opacity. If ln = True, then T and rho are
    lnT and lnrho. Otherwise we convert them.
    
     Parameters
     ----------
     T : float,
         Temperature in [cgs].
     rho : float,
         Density in [cgs].
     kind : str,
         The kind of opacities. Valid choices are:
         rosseland, plank or effective.
     log : bool,
         If True, then T and rho are lnT and lnrho, Default is True
     
    Returns
    -------
    opacity : float,
        The rosseland mean opacity in [cgs].
    '''    
    if not ln: 
        rho = np.log(rho)
        # Remove fuckery
        rho = np.nan_to_num(rho, nan = 0, posinf = 0, neginf= 0)
    
    # Pick Opacity & Use Interpolation Function
    if kind == 'rosseland':
        ln_opacity = natural_ross(rho)

    elif kind == 'planck':
        ln_opacity = natural_planck((rho))
        
    elif kind == 'effective':
        planck = natural_planck(rho)
        scattering = natural_scatter((rho))
        
        # Rybicky & Lightman eq. 1.98
        ln_opacity = np.sqrt(planck * (planck + scattering)) 
    else:
        print('Invalid opacity type. Try: rosseland / planck / effective.')
        return 1
            
    # Remove the ln
    opacity = np.exp(ln_opacity)

    return opacity

print(natural_ross(90))
    
