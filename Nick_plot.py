import numpy as np
import matplotlib.pyplot as plt

days6 = [1, 1.14, 1.3, 1.4]
L6 = np.loadtxt('L_m6.txt')

days4 = [1, 1.1, 1.2, 1.3, 1.57, 1.7, 1.83]
L4 = np.loadtxt('L_m4.txt')

plt.figure(figsize=(15,8))
plt.plot(days6, L6, 'o-', label = r'$M_{BH}=10^6M_{sun}$', c = 'olivedrab')
plt.plot(days4, L4, 'o-', label = r'$M_{BH}=10^4M_{sun}$', c = 'slateblue')
plt.ylabel(r'$log_{10}$ Luminosity [erg/s]')
plt.xlabel(r'$t/t_{fb}$')
plt.grid()
plt.yscale('log')
plt.legend()
plt.savefig('Bolom46.png')
plt.show()