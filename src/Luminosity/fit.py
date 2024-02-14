""""
@author: paola , konstantinos
"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import src.Utilities.prelude as c
import src.Utilities.selectors as s

##
# VARIABLES
##

m = 6
# ztf: 3.23e14 - 6.71e14 // swift: 6.34e14 - 1.88e15
freq_min = 3.23e14
freq_max = 1.88e15


##
# FUNCTION
##s
def tofit(n, R, T):
    const = 2*c.h/c.c**2
    planck = const * n**3 / (np.exp(c.h*n/(c.Kb*T))-1)
    Lum = 4 * np.pi**2 * R**2 * planck 
    return Lum

##
# MAIN
##

if __name__ == '__main__':
    plot = False
    save = True
    do = True
    check = 'fid'

    # Load & Unpack
    snapshots, days = s.select_snap(m, check)

    x = np.loadtxt(f'data/blue/spectrafreq_m{m}.txt') # x = logν

    freqs = np.power(10, x)
    init_R = 5e14
    init_T = 2e4
    
    freq_min_idx = np.argmin( np.abs(freqs - freq_min))
    freq_max_idx = np.argmin( np.abs(freqs - freq_max))

    fit_freqs = freqs[freq_min_idx:freq_max_idx]
    blue_avg = np.zeros(len(snapshots))

    for idx_snap in range(len(snapshots)):
        snap = snapshots[idx_snap]
        print(snap)
        allspectra = np.loadtxt(f'data/blue/allnLn_single_m{m}_{snap}.txt')
        temp = np.zeros(len(allspectra))
        radius = np.zeros(len(allspectra))
        Blue = np.zeros(len(allspectra))

        # loop over the spectrum of each observer
        # NOTE: last 4 because we save on top all the time and should fix that
        for idx_obs in range(len(allspectra)):#(len(data) - 4, len(data)):
            Lums = allspectra[idx_obs]
            Lums_fit = Lums[freq_min_idx:freq_max_idx]
            fit = curve_fit(tofit, fit_freqs, Lums_fit, p0 = (init_R, init_T))

            # # Integrate in log10: better to divide in the 3 bands to integrate
            # shouldn't we consider all the frequencies?
            # fitted = tofit(freqs, fit[0][0], fit[0][1] )
            # x_fitted =  freqs * fitted
            # b = np.trapz(x_fitted, x) 
            # b *= np.log(10)
            b = 4 * np.pi * (fit[0][0])**2 * c.sigma * (fit[0][1])**4
            temp[idx_obs] = fit[0][1]
            radius[idx_obs] = fit[0][0]
            Blue[idx_obs]= b

        blue_avg[idx_snap] = np.mean(Blue)    
        if save:
            with open(f'data/blue/singlebluedata_m{m}_{snap}.txt', 'w') as f:
                f.write('# Fitted quantities for snapshots '+ str(snap) + '\n#Temperature \n')
                f.write(' '.join(map(str, temp)) + '\n')
                f.write('# Radius \n')
                f.write(' '.join(map(str, radius)) + '\n')
                f.write('# Bolometric L \n')
                f.write(' '.join(map(str, Blue)) + '\n')
                f.close()
            
    if save:    
        with open(f'data/blue/blue_m{m}.txt', 'w') as fileblue:
            fileblue.write('# Bolometric L for each snapshot\n# t/t_fb\n')
            fileblue.write(' '.join(map(str, days)) + '\n')
            fileblue.write('# Bolometric L\n')
            fileblue.write(' '.join(map(str, blue_avg)) + '\n')
            fileblue.close()
    
    # OLD CODE            
    # if plot:
    #     fig, axs = plt.subplots(2,3, tight_layout = True)
    #     axs2 = []
    #     for i in range(2):
    #         for j in range(3):
    #             axs2.append(axs[i,j])
              
    #     for i, ax in enumerate(axs2):
    #         if i == 5:
    #             break
    #         Lums = data[i] #data[i + len(data) - 4] # NOTE: last 4 because we save on top all the time and should fix that
    #         Lums_fit = Lums[freq_min_idx:freq_max_idx]
    #         fit = curve_fit(tofit, fit_freqs, Lums_fit, p0 = (init_R, init_T))
            
    #         # Plot
    #         ax.scatter(fit_freqs, Lums_fit, c = 'coral', label = 'Fititng points', s = 4)
    #         fitted = tofit(freqs, fit[0][0], fit[0][1] )
    #         ax.plot(freqs, fitted,
    #                   color = 'limegreen', label = 'Fitted')
    #         ax.plot(freqs, Lums,
    #                  color = 'royalblue', linestyle = 'dashed', label = 'Spectrum')
            
    #         trial = [ tofit(n, init_R, init_T ) for n in freqs ]
    #         ax.plot(freqs, trial,
    #                  color = 'grey', label = 'Initial Guess', linestyle = '--')
    #         ax.grid()
    #         ax.set_xscale('log')
    #         ax.set_yscale('log')
    #         ax.set_xlabel(r'$\nu$ [Hz]')
    #         ax.set_ylabel(r'$L_\nu$')
    #         ax.legend(fontsize = 4)
    #         ax.set_ylim(1e17,1e30)
    #     if save: 
    #         plt.savefig('Final_plot/Fit_m' + str(m) + '.png')
    #     plt.show()