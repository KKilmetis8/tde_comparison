import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [10 , 4]

import sys
sys.path.append('/Users/paolamartire/tde_comparison')

m = 6
c = 2.99792458e10 #[cm/s]
snap = 881

nL_tilde_n = np.loadtxt(f'data/blue/nLn_single_m{m}.txt')
x_array = np.loadtxt(f'data/blue/spectrafreq_m{m}.txt')
n_array = np.power(10, x_array)
n_start = 1e13
n_end = 1e18
wavelength = np.divide(c, n_array) * 1e8 # A

fig, ax1 = plt.subplots( figsize = (6,6) )
ax1.plot(n_array, n_array * nL_tilde_n[1], c = 'r', label = r'$-\vec{x}$')
ax1.plot(n_array, n_array * nL_tilde_n[0], linestyle = 'dashed', c = 'b', label = r'$\vec{x}$')
ax1.plot(n_array, n_array * nL_tilde_n[2], c = 'magenta', label = r'$\vec{z}$')
ax2 = ax1.twiny()
ax1.set_xlabel(r'$log_{10}\nu$ [Hz]')
ax1.set_ylabel(r'$log_{10}(\nu L_\nu)$ [erg/s]')
ax1.set_ylim(1e39, 2e42)
ax1.set_xlim(n_start,n_end)
ax1.loglog()
ax1.grid()
#ax2.plot(wavelength, n_array * nL_tilde_n[0])
ax2.set_xlim(c/n_end *1e8, c/n_start * 1e8)
ax2.invert_xaxis()
ax2.loglog()
ax2.set_xlabel(r'Wavelength [\AA]')
ax1.legend()
plt.savefig(f'Figs/singlespectra{snap}')
plt.show()



        # if telescope: 
        #     ultrasat_min = 1.03e15
        #     ultrasat_max = 1.3e15
        #     r_min = 4.11e14
        #     r_max = 5.07e14
        #     g_min = 5.66e14
        #     g_max = 7.48e14

        #     plt.xlim(14,16)
        #     plt.ylim(1e22,1e30)
        #     plt.axvline(np.log10(ultrasat_min), color = 'b')
        #     plt.axvline(np.log10(ultrasat_max), color = 'b')
        #     plt.axvspan(np.log10(ultrasat_min), np.log10(ultrasat_max), alpha=0.4, color = 'b')
        #     plt.text(np.log10(ultrasat_min)+0.04,1e23,'ULTRASAT', rotation = 90)

        #     plt.axvline(np.log10(r_min), color = 'r')
        #     plt.axvline(np.log10(r_max), color = 'r')
        #     plt.axvspan(np.log10(r_min), np.log10(r_max), alpha=0.4, color = 'r')
        #     plt.text(np.log10(r_min)+0.04,1e23,'R-band ZTF', rotation = 90)

        #     plt.axvline(np.log10(g_min), color = 'orange')
        #     plt.axvline(np.log10(g_max), color = 'orange')
        #     plt.axvspan(np.log10(g_min), np.log10(g_max), alpha=0.4, color = 'orange')
        #     plt.text(np.log10(g_min)+0.05,1e23,'G-band ZTF', rotation = 90)
        #     plt.legend()
        #     plt.savefig('telescope_spectra_m' + str(m) + '.png' )

