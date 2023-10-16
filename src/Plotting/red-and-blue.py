"""
Let's do blue-red curve of 10^4 and 10^6 BH

author: paola

"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

import matplotlib.pyplot as plt
import numpy as np

# red_4 = np.loadtxt('data/reddata_m4.txt')
# blue_4 = np.loadtxt('data/bluedata_m4.txt')
red_6 = np.loadtxt('data/reddata_m6.txt')
blue_6 = np.loadtxt('data/bluedata_m6.txt')

# red_4_days = red_4[0]
# red_4_lum = np.log10(red_4[1])

# blue_4_days = blue_4[0]
# blue_4_lum = np.log10(blue_4[1])


red_6_days = red_6[0]
red_6_lum = np.log10(red_6[1])

blue_6_days = blue_6[0]
blue_6_lum = np.log10(blue_6[1])

# plt.plot(red_4_days, red_4_lum, "o-", color = 'coral', label = "FLD $10^4M_{sun}$", linestyle = 'dashed')
# plt.plot(blue_4_days, blue_4_lum, "o-", color = 'deepskyblue', label = "BB $10^4M_{sun}$", linestyle = 'dashed')
plt.plot(red_6_days, red_6_lum, "o-", color = 'coral', label = "FLD $10^6M_{sun}$")
plt.plot(blue_6_days, blue_6_lum, "o-", color = 'deepskyblue', label = "BB $10^6M_{sun}$")
plt.ylabel("$log_{10}$Luminosity", fontsize = 10)
plt.xlabel("$t/t_{fb}$", fontsize = 10)
plt.xlim(1,1.4)
plt.legend(loc = 'lower right', fontsize = 8)
plt.savefig('multiplot.png')
plt.grid()
plt.show()
