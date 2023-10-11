""""
@author: paola 
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.figsize'] = [10 , 8]
plt.rcParams['axes.facecolor']= 	'whitesmoke'
AEK = '#F1C410'

from scipy.optimize import curve_fit

m = 6
c = 2.99792458e10 # [cm/s]
h = 6.62607015e-27 # [gcm^2/s]
Kb = 1.380649e-16 #[gcm^2/s^2K]

def tofit(n, R, T):
    const = 2*h/c**2
    planck = const * n**3 / (np.exp(h*n/(Kb*T))-1)
    L = 4*np.pi * R**2 * planck 
    return L

#
UV_min = 1e12
UV_max = 1e20

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
    
    UV_min_idx = np.argmin( np.abs(freqs - UV_min))
    UV_max_idx = np.argmin( np.abs(freqs - UV_max))
    freqs = freqs[UV_min_idx:UV_max_idx]
    x = x[UV_min_idx:UV_max_idx]
    
    if do:
        Blue = []
        for i in range(1, 3):
            Lums = data[i] 
            Lums = Lums[UV_min_idx:UV_max_idx]
            fit = curve_fit(tofit, freqs, Lums,
            p0 = (init_R, init_T))
            fitted = [ tofit(n, fit[0][0], fit[0][1]) for n in freqs]
            
            # Integrate in log10
            x_fitted =  freqs * fitted

            b = np.trapz(x_fitted, x) 
            b *= np.log(10)
            Blue.append(b)
            
        if save:
           np.save('data/bluedata_m'+ str(m), Blue) 
                
    if plot:
        fig, axs = plt.subplots(2,2, tight_layout = True)
        axs2 = []
        for i in range(2):
            for j in range(2):
                axs2.append(axs[i,j])
              
        for i, ax in enumerate(axs2):
            i += 1
            Lums = data[i] # First snapshot
            Lums = Lums[UV_min_idx:UV_max_idx]
            # Fit
            fit = curve_fit(tofit, freqs, Lums,
                            p0 = (init_R, init_T))
            
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
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel(r'$L_\nu$')
            #ax.legend()
            ax.set_ylim(1e19,1e28)