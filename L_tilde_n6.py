import numpy as np
import matplotlib.pyplot as plt

m = 6
L_tilda_n = np.loadtxt('Ltilda_m'+ str(m) + '.txt')

n_logspace = np.loadtxt('Ltilda_m'+ str(m) + '.txt')[0]
L_tilda_n844 = L_tilda_n[1]
L_tilda_n881 = L_tilda_n[2]
L_tilda_n925 = L_tilda_n[3]
L_tilda_n950 = L_tilda_n[4]

plt.plot(n_logspace, L_tilda_n844, c = 'tomato', label = 't/tf = 1')
plt.plot(n_logspace, L_tilda_n881, c = 'orange', label = 't/tf = 1.14')
plt.plot(n_logspace, L_tilda_n925, c = 'yellowgreen', label = 't/tf = 1.3')
plt.plot(n_logspace, L_tilda_n950, c = 'teal', label = 't/tf = 1.4')

plt.xlabel(r'$log\nu$ [Hz]')
plt.xlim(12,18)
plt.ylim(1e38, 1e45)
plt.yscale('log')
plt.ylabel(r'$log\tilde{L}_\nu$ [erg/s]')
plt.grid()
plt.savefig('TOT_Ltilda_m' + str(m) )
plt.show()
plt.legend()
plt.axvline(15, color = 'tab:orange')
plt.axvline(17, color = 'tab:orange')
plt.axvspan(15, 17, alpha=0.5, color = 'tab:orange')

