""""
@author: paola 
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.figsize'] = [5 , 4]
plt.rcParams['axes.facecolor']= 	'whitesmoke'
AEK = '#F1C410'

from scipy.optimize import curve_fit


m = 6
c = 2.99792458e10 # [cm/s]
h = 6.62607015e-27 # [gcm^2/s]
Kb = 1.380649e-16 #[gcm^2/s^2K]

def tofit(n, R, T):
    const = 2*h/c**2
    plank = const * n**3 / (np.exp(h*n/(Kb*T))-1)
    L = 4 * np.pi * R**2 * plank 
    return L

# Load & Unpack
path = 'Final-plot/Data/'
data = np.loadtxt(path + 'L_tilde_n_m' + str(m) + '.txt')
a = data[0] # a = logν
freqs = np.power(10, a)
Lums = data[1] # First snapshot

# Fit
fit = curve_fit(tofit, freqs, Lums,
                p0 = (1e14, 6000))

# Plot
fitted = [ tofit(n, fit[0][0], fit[0][1]) for n in freqs]
plt.plot(freqs, fitted,
          color = AEK, label = 'Fitted')
plt.plot(freqs, Lums,
         color = 'k', linestyle = 'dashed', label = 'Target')

trial = [ tofit(n, 1e14, 3e4 ) for n in freqs ]
plt.plot(freqs, trial,
         color = 'cadetblue', label = 'Initial Guess')
plt.grid()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Frequency [Hz]')
plt.ylabel(r'$L_\nu$')
plt.legend()
plt.ylim(1e21,1e27)