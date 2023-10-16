""""
@author: paola , konstantinos
"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.figsize'] = [10 , 8]
plt.rcParams['axes.facecolor']= 	'whitesmoke'
AEK = '#F1C410'

##
# VARIABLES
##

m = 6
c = 2.99792458e10 # [cm/s]
h = 6.62607015e-27 # [gcm^2/s]
Kb = 1.380649e-16 #[gcm^2/s^2K]
Rztf_min = 4.11e14
Rztf_max = 5.07e14
Gztf_min = 5.66e14
Gztf_max = 7.48e14
ultrasat_min = 1.03e15
ultrasat_max = 1.3e25

##
# FUNCTION
##
def tofit(n, R, T):
    const = 2*h/c**2
    planck = const * n**3 / (np.exp(h*n/(Kb*T))-1)
    Lum = 4 * np.pi**2 * R**2 * planck 
    return Lum

##
# MAIN
##

if __name__ == '__main__':
    plot = True
    save = True
    do = True

    # Load & Unpack
    path = 'data/'
    data = np.loadtxt(path + 'L_spectrum_m' + str(m) + '.txt')
    x = data[0] # x = logν

    freqs = np.power(10, x)
    init_R = 1e12
    init_T = 3e6
    
    Rztf_min_idx = np.argmin( np.abs(freqs - Rztf_min))
    Rztf_max_idx = np.argmin( np.abs(freqs - Rztf_max))
    Gztf_min_idx = np.argmin( np.abs(freqs - Gztf_min))
    Gztf_max_idx = np.argmin( np.abs(freqs - Gztf_max))
    ultrasat_min_idx = np.argmin( np.abs(freqs - ultrasat_min))
    ultrasat_max_idx = np.argmin( np.abs(freqs - ultrasat_max))

    Rztf_freqs = freqs[Rztf_min_idx:Rztf_max_idx]
    Gztf_freqs = freqs[Gztf_min_idx:Gztf_max_idx]
    ultrasat_freqs = freqs[ultrasat_min_idx:ultrasat_max_idx]
    fit_freqs = np.concatenate((Rztf_freqs, Gztf_freqs, ultrasat_freqs))
    # Rztf_x = x[Rztf_min_idx:Rztf_max_idx]
    # Gztf_x = x[Gztf_min_idx:Gztf_max_idx]
    # ultrasat_x = x[ultrasat_min_idx:ultrasat_max_idx]
    #fit_x = np.concatenate((Rztf_x, Gztf_x, ultrasat_x))

    if do:
        Blue = []
        for i in range(1 , len(data)):
            Lums = data[i] 
            # print(Lums)
            Rztf_Lums = Lums[Rztf_min_idx:Rztf_max_idx]
            Gztf_Lums = Lums[Gztf_min_idx:Gztf_max_idx]
            ultrasat_Lums = Lums[ultrasat_min_idx:ultrasat_max_idx]
            Lums_fit = np.concatenate((Rztf_Lums, Gztf_Lums, ultrasat_Lums))
            fit = curve_fit(tofit, fit_freqs, Lums_fit, p0 = (init_R, init_T))

            # # Integrate in log10: better to divide in the 3 bands to integrate
            # shouldn't we consider all the frequencies?
            # fitted = [tofit(n, fit[0][0], fit[0][1]) for n in freqs]
            # x_fitted =  freqs * fitted
            # b = np.trapz(x_fitted, x) 
            # b *= np.log(10)

            fitted = [tofit(n, fit[0][0], fit[0][1]) for n in freqs]
            x_fitted =  freqs * fitted
            b = np.trapz(x_fitted, x) 
            b *= np.log(10)

            Blue.append(b)
            
        if save:
           np.savetxt('data/bluedata_m' + str(m) + '.txt', Blue) 
                
    if plot:
        fig, axs = plt.subplots(2,2, tight_layout = True)
        axs2 = []
        for i in range(2):
            for j in range(2):
                axs2.append(axs[i,j])
              
        for i, ax in enumerate(axs2):
            i += 1
            Lums = data[i] # First snapshot
            Rztf_Lums = Lums[Rztf_min_idx:Rztf_max_idx]
            Gztf_Lums = Lums[Gztf_min_idx:Gztf_max_idx]
            ultrasat_Lums = Lums[ultrasat_min_idx:ultrasat_max_idx]
            Lums_fit = np.concatenate((Rztf_Lums, Gztf_Lums, ultrasat_Lums))
            fit = curve_fit(tofit, fit_freqs, Lums_fit, p0 = (init_R, init_T))
            
            # Plot
            fitted = [ tofit(n, fit[0][0], fit[0][1]) for n in freqs]
            ax.plot(freqs, fitted,
                      color = AEK, label = 'Fitted')
            ax.plot(freqs, Lums,
                     color = 'k', linestyle = 'dashed', label = 'Target')
            
            trial = [ tofit(n, init_R, init_T ) for n in freqs ]
            ax.plot(freqs, trial,
                     color = 'cadetblue', label = 'Initial Guess')
            ax.grid()
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r'$\nu$ [Hz]')
            ax.set_ylabel(r'$L_\nu$')
            ax.legend(fontsize = 6)
            ax.set_ylim(1e19,1e30)
        plt.show()