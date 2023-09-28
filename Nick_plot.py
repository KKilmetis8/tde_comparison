import numpy as np
import matplotlib.pyplot as plt

days6 = [1, 1.14, 1.3, 1.4]
L6 = np.loadtxt('L_m6.txt')

days4 = [1, 1.2, 1.3]
L4 = np.loadtxt('L_m4.txt')

plt.plot(days6, L6, 'o-', label = r'$M^6 M_{sun}$')
plt.plot(days4, L4, 'o-', label = r'$M^4 M_{sun}$')
plt.ylabel(r'$log_{10}$ Luminosity [erg/s]')
plt.xlabel(r'$t/t_{fb}$')
plt.grid()
plt.yscale('log')
plt.legend()
plt.savefig('Nick.png' )
plt.show()