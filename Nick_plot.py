import numpy as np
import matplotlib.pyplot as plt


n_min = 1e12 
n_max = 1e19
n_spacing = 100
n_array = np.linspace(n_min, n_max, num = n_spacing)
n_logspace = np.linspace(np.log10(n_min), np.log10(n_max), num = n_spacing)

# CHECK
# for i in range(1,len(n_array)-1):
#     print(n_array[i]-n_array[i-1])

# print('------')

# for i in range(1,len(n_array)-1):
#     print(n_logspace[i]-n_logspace[i-1])

days6 = np.loadtxt('L6.txt')[0]
L6 = np.loadtxt('L6.txt')[1]

days4 = np.loadtxt('L4.txt')[0]
L4 = np.loadtxt('L4.txt')[1]

plt.plot(days6, L6, 'o-', label = r'$M^6 M_{sun}$')
plt.plot(days4, L4, 'o-', label = r'$M^4 M_{sun}$')
plt.ylabel(r'$log_{10}$ Luminosity [erg/s]')
plt.xlabel(r'$t/t_{fb}$')
plt.grid()
plt.yscale('log')
plt.legend()
plt.show()