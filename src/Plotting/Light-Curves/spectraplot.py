import sys
sys.path.append('/Users/paolamartire/tde_comparison')
import scipy.io

import numpy as np
import matplotlib.pyplot as plt
import src.Utilities.prelude as c
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [3 , 3]
plt.rc('xtick', labelsize = 15) 
plt.rc('ytick', labelsize = 15) 

m = 4
snap = 293
num = 1000
pre = 'dot'
opacity = 'cloudy'
axis = 'temp'

def temperature(n):
        return n * c.h / c.Kb

def frequencies(T):
        return T * c.Kb / c.h

def wavelength(n):
        # in angststrom 
        return c.c *1e8 / n 

# x axis 
x_array = np.loadtxt(f'data/blue/spectrafreq_m{m}.txt')
n_array = np.power(10, x_array)
T_start = 1e3
T_end = 1e8
n_start = frequencies(T_start)
n_end = frequencies(T_end)
lamda = wavelength(n_array)

# y axis 
nL_tilde_n = np.loadtxt(f'data/blue/{opacity}_nLn_single_m{m}_{snap}_1000.txt')
nL_tilde_n *=1.6339520760162161/9.763337052609916

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

fig, ax1 = plt.subplots( figsize = (9,6) ) 
ax1.plot(x_axis, n_array * nL_tilde_n[0], c = 'b',  label = r'$\vec{x}$')
ax1.plot(x_axis, n_array * nL_tilde_n[0], c = 'b',  label = r'$\vec{x}$')
ax1.plot(x_axis, n_array * nL_tilde_n[1], c = 'r', label = r'$-\vec{x}$')
ax1.plot(x_axis, n_array * nL_tilde_n[2], c = 'k', label = r'$\vec{y}$')
ax1.plot(x_axis, n_array * nL_tilde_n[3], c = 'lime', label = r'$-\vec{y}$')
ax1.plot(x_axis, n_array * nL_tilde_n[4], c = 'magenta', label = r'$\vec{z}$')
ax1.plot(x_axis, n_array * nL_tilde_n[5], c = 'aqua', label = r'$-\vec{z}$')
ax2 = ax1.twiny()
ax1.set_xlabel(f'{label}', fontsize = 16)
ax1.set_ylabel(r'$log_{10}(\nu L_\nu)$ [erg/s]', fontsize = 16)
if m == 4:
        y_lowlim = 1e36
        y_highlim = 1e42
else:
        y_lowlim = 2e40
        y_highlim = 1.3e45
ax1.set_xlim(x_start,x_end)
ax1.set_ylim(y_lowlim, y_highlim)
ax1.loglog()
ax1.grid()
ax1.legend()
ax1.set_title(f'Spectra {snap} with {opacity} opacity')



ax2.set_xlim(wavelength(n_start), wavelength(n_end))
ax2.plot(wavelength(n_array), n_array * nL_tilde_n[0],  c = 'b')
ax2.set_xlim(c.c/n_end *1e8, c.c/n_start * 1e8)
ax2.invert_xaxis()
ax2.loglog()
ax2.set_xlabel(r'$log_{10}\lambda [\AA]$', fontsize = 16)

# ax2.axvline(120, c = 'mediumorchid')
# ax2.axvspan(120, 4000, color = 'mediumorchid', alpha = 0.4)
# ax2.text(160, y_highlim/2, 'UV', rotation = 90, fontsize = 10)

# ax2.axvline(4000, c = 'gold')
# ax2.axvspan(4000, 7000, color = 'gold', alpha = 0.4)
# ax2.axvline(7000, c = 'gold', label = 'visible')
# ax2.text(6000, y_highlim/5, 'visible', rotation = 90, fontsize = 10)

plt.savefig(f'Figs/TEST{opacity}_spectra{snap}.png')
plt.show()

# Elad
# plt.figure()
# import mat73
# mat = mat73.loadmat('data/data_308.mat')
# elad_T = np.array([ temperature(n) for n in mat['nu']])
# #elad_T = np.logspace(3,13,1000)[::10]
# for obs in range(1):
#     y = np.multiply(mat['nu'], 1)
#     y = np.multiply(y, mat['F_photo'][obs])
#     plt.loglog(elad_T, y, c='k', linestyle = '--', label ='One of Elads obs')

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

