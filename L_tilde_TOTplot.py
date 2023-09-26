import numpy as np
import matplotlib.pyplot as plt

m = 4
L_tilda_n = np.loadtxt('Ltilda_m'+ str(m) + '.txt')

n_logspace = L_tilda_n[0]

if m == 6:
    L_tilda_n844 = L_tilda_n[1]
    L_tilda_n881 = L_tilda_n[2]
    L_tilda_n925 = L_tilda_n[3]
    L_tilda_n950 = L_tilda_n[4]

    plt.plot(n_logspace, L_tilda_n844, c = 'tomato', label = 't/$t_{fb}$ = 1')
    plt.plot(n_logspace, L_tilda_n881, c = 'orange', label = 't/$t_{fb}$ = 1.14')
    plt.plot(n_logspace, L_tilda_n925, c = 'yellowgreen', label = 't/$t_{fb}$ = 1.3')
    plt.plot(n_logspace, L_tilda_n950, c = 'teal', label = 't/$t_{fb}$ = 1.4')

    plt.xlim(12,19)
    plt.ylim(1e35, 1e45)

if m == 4:
    L_tilda_n233 = L_tilda_n[1]
    L_tilda_n254 = L_tilda_n[2]
    L_tilda_n263 = L_tilda_n[3]
    #L_tilda_n950 = L_tilda_n[4]

    plt.plot(n_logspace, L_tilda_n233, c = 'tomato', label = 't/$t_{fb}$ = 1')
    plt.plot(n_logspace, L_tilda_n254, c = 'orange', label = 't/$t_{fb}$ = 1.2')
    plt.plot(n_logspace, L_tilda_n263, c = 'yellowgreen', label = 't/$t_{fb}$ = 1.3')
    #plt.plot(n_logspace, L_tilda_n263, c = 'teal', label = 't/$t_{fb}$ = 1.4')

    plt.xlim(12,18)
    plt.ylim(1e32, 1e42)

plt.xlabel(r'$log\nu$ [Hz]')
plt.yscale('log')
plt.ylabel(r'$log\tilde{L}_\nu$ [erg/s]')
plt.grid()
plt.legend()
#plt.axvline(14, color = 'tab:orange')
#plt.axvline(17, color = 'tab:orange')
#plt.axvspan(14, 17, alpha=0.3, color = 'tab:orange', label = 'UV band')
plt.savefig('TOT_Ltilda_m' + str(m) + '.png' )
plt.show()

