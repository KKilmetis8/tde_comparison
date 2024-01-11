import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('/Users/paolamartire/tde_comparison')

m = 6
snap = 881
axis = 'temp'

c = 2.99792458e10 #[cm/s]
h = 6.62607015e-27 #[gcm^2/s]
Kb = 1.380649e-16 #[gcm^2/s^2K]

def temperature(n):
        return n * h / Kb

def frequencies(T):
        return T * Kb / h

def wavelength(n):
        # in angststrom 
        return c *1e8 / n 

# x axis 
x_array = np.loadtxt(f'data/blue/spectrafreq_m{m}.txt')
n_array = np.power(10, x_array)
T_start = 1e3
T_end = 1e8
n_start = frequencies(T_start)
n_end = frequencies(T_end)
lamda = wavelength(n_array)

# y axis 
#nL_tilde_n = np.loadtxt(f'data/blue/nLn_single_m{m}_{snap}.txt')
nL_tilde_n = np.loadtxt(f'data/blue/TESTobserv_nLn_single_m{m}_{snap}.txt')

if axis == 'freq':
        x_axis = n_array
        x_start = n_start
        x_end = n_end
        label = r'$log_{10}\nu$ [Hz]'
if axis == 'temp':
        x_axis = temperature(n_array)
        label = r'$log_{10}$T [K]'
        x_start = T_start
        x_end = T_end

fig, ax1 = plt.subplots( figsize = (8,6) ) 
ax1.plot(x_axis, n_array * nL_tilde_n[0], c = 'b',  label = r'$\vec{x}$')
ax1.plot(x_axis, n_array * nL_tilde_n[1], c = 'r', label = r'$-\vec{x}$')
ax1.plot(x_axis, n_array * nL_tilde_n[2], c = 'k', label = r'$\vec{y}$')
ax1.plot(x_axis, n_array * nL_tilde_n[3], c = 'lime', label = r'$-\vec{y}$')
ax1.plot(x_axis, n_array * nL_tilde_n[4], c = 'magenta', label = r'$\vec{z}$')
ax1.plot(x_axis, n_array * nL_tilde_n[5], c = 'aqua', label = r'$-\vec{z}$')
ax2 = ax1.twiny()
ax1.set_xlabel(f'{label}')
ax1.set_ylabel(r'$log_{10}(\nu L_\nu)$ [erg/s]')
ax1.set_ylim(2e39, 1e44)
ax1.set_xlim(x_start,x_end)
ax2.set_xlim(wavelength(n_start),wavelength(n_end))
ax1.loglog()
ax1.grid()
ax2.plot(wavelength(n_array), n_array * nL_tilde_n[0],  c = 'b')
ax2.set_xlim(c/n_end *1e8, c/n_start * 1e8)
ax2.invert_xaxis()
ax2.loglog()
ax2.set_xlabel(r'$log_{10}\lambda [\AA]$')
ax1.legend()
ax1.set_title(r'Spectrum $\vec{x}$')
plt.savefig(f'Figs/test_spectra{snap}')
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

