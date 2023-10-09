"""
Created on Mon Oct 9 2023

@author: konstantinos, paola 

Calculate the luminosity that we will use in the blue (BB) curve as sum of multiple BB.

NOTES FOR OTHERS:
- All the functions have to be applied to a CELL
- arguments are in cgs, NOT in log.
- make changes in VARIABLES: frequencies range, fixes (number of snapshots) anf thus days
"""

# Vanilla imports
import numpy as np
import matplotlib.pyplot as plt

# Chocolate imports
from src.Luminosity.photosphere import get_photosphere
from src.Optical_Depth.opacity_table import opacity

# Constants
c = 2.99792458e10 #[cm/s]
h = 6.62607015e-27 #[gcm^2/s]
Kb = 1.380649e-16 #[gcm^2/s^2K]
alpha = 7.5646 * 10**(-15) # radiation density [erg/cm^3K^4]
Rsol_to_cm = 6.957e10

###
# FUNCTIONS
###

def log_array(n_min, n_max, lenght):
    x_min = np.log10(n_min)
    x_max = np.log10(n_max)
    x_arr = np.linspace(x_min, x_max , num = lenght)
    return x_arr

def select_fix(m):
    if m == 4:
        snapshots = [233, 254, 263, 277 , 293, 308, 322]
        days = [1, 1.2, 1.3, 1.4, 1.56, 1.7, 1.8] 
    if m == 6:
        snapshots = [844, 881, 925, 950]
        days = [1, 1.1, 1.3, 1.4] #t/t_fb
    return snapshots, days

def planck(Temperature: float, n: float) -> float:
    """ Planck function in a cell. It needs temperature and frequency. """
    
    const = 2*h/c**2
    fun = const * n**3 / (np.exp(h*n/(Kb*Temperature))-1)
    return fun

def luminosity_n(Temperature: float, Density: float, tau: float, volume: float, n:int):
    """ Luminosity in a cell: L_ni = \epsilon e^(-\tau) B_ni / B where  
    B = \sigma T^4/\pi"""
    
    k_planck = opacity(Temperature, Density, 'planck', ln = False)
    L = 4 * np.pi * k_planck * volume * np.exp(-tau) * planck(Temperature, n)
    return L

def normalisation(L_x: np.array, x_array: np.array, luminosity_fld: float) -> float:
    """ Given the array of luminosity L_x computed over 10^{x_array} (!!!), 
    find the normalisation constant from FLD model used for L_tilde_nu. """  
    
    xLx =  10**(x_array) * L_x
    L = np.trapz(xLx, x_array) 
    L *= np.log(10)
    norm = luminosity_fld / L
    return norm

# MAIN
if __name__ == "__main__":
    m = 4
    n_min = 1e12 
    n_max = 1e18
    n_spacing = 10000
    x_arr = log_array(n_min, n_max, n_spacing)
    snap_index = 0

    snapshots, days = select_fix(m)
    fld_data = np.loadtxt('reddata_m'+ str(m) +'.txt')
    fix = snapshots[snap_index]
    luminosity_fld_fix = fld_data[1]
    n_arr = 10**x_arr

    rays_den, rays_T, rays_tau, photosphere, radii = get_photosphere(fix, m)
    dr = (radii[1] - radii[0]) * Rsol_to_cm
    volume = 4 * np.pi * radii**2 * dr  / 192

    lum_n = np.zeros(len(x_arr))
    for j in range(len(rays_den)):
        for i in range(len(rays_tau[j])):        
            # Temperature, Density and volume: np.array from near to the BH to far away. 
            # Thus we will use negative index in the for loop.
            # tau: np.array from outside to inside.      
            T = rays_T[j][-i]
            rho = rays_den[j][-i] 
            opt_depth = rays_tau[j][i]
            cell_vol = volume[-i]

            # Ensure we can interpolate
            rho_low = np.exp(-22)
            T_low = np.exp(8.77)
            T_high = np.exp(17.8)
            if rho < rho_low or T < T_low or T > T_high:
                continue
            
            for x_index in range(len(x_arr)): #we need linearspace
                freq = 10**x_arr[x_index]
                lum_n_cell = luminosity_n(T, rho, opt_depth, cell_vol, freq)
                lum_n[x_index] += lum_n_cell
        
        print('ray:', j)
    
    # Normalise with the bolometric luminosity from red curve (FLD)
    const_norm = normalisation(lum_n, x_arr, luminosity_fld_fix[snap_index])
    lum_tilde_n = lum_n * const_norm

    # Find the bolometic energy (should be = to the one from FLD)
    bolom_integrand =  n_arr * lum_tilde_n
    bolom = np.log(10) * np.trapz(bolom_integrand, x_arr)
    bolom = "{:.4e}".format(bolom) #scientific notation
    print('bolometric L:', bolom)

    # Save data and plot
    with open('L_m' + str(m) + '.txt', 'a') as fbolo:
        fbolo.write('#snap '+ str(fix) + '\n')
        fbolo.write(bolom + '\n')
        fbolo.close()

    fig, ax = plt.subplots()
    ax.plot(n_arr, lum_tilde_n)
    plt.xlabel(r'$log_{10}\nu$ [Hz]')
    plt.ylabel(r'$log_{10}\tilde{L}_\nu$ [erg/sHz]')
    plt.loglog()
    plt.grid()
    plt.text(0.7e12, 2e23, r'$t/t_{fb}:$ ' + f'{days[snap_index]}\n B: {bolom}')
    plt.savefig('Ltildan_m' + str(m) + '_snap' + str(fix))
    plt.show()

    plt.figure()
    plt.plot(n_arr, n_arr * lum_tilde_n)
    plt.xlabel(r'$log_{10}\nu$ [Hz]')
    plt.ylabel(r'$log_{10}(\nu\tilde{L}_\nu)$ [erg/s]')
    plt.loglog()
    plt.grid()
    plt.savefig('n_Ltildan_m' + str(m) + '_snap' + str(fix))
    plt.show()