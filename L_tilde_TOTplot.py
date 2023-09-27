import numpy as np
import matplotlib.pyplot as plt

m = 6
plot = 'bolometric'
L_tilda_n = np.loadtxt('Ltilda_m'+ str(m) + '.txt')
n_array = np.linspace(1e12, 1e19, num = 100)
n3 = n_array**3
print(n3)
n4 = n_array**4

n_logspace = L_tilda_n[0]

if m == 6:
    L_tilda_n844 = L_tilda_n[1]
    L_tilda_n881 = L_tilda_n[2]
    L_tilda_n925 = L_tilda_n[3]
    L_tilda_n950 = L_tilda_n[4]

    L844 = np.trapz(L_tilda_n844, n_array)
    L881 = np.trapz(L_tilda_n881, n_array)
    L925 = np.trapz(L_tilda_n925, n_array)
    L950 = np.trapz(L_tilda_n950, n_array)

    if plot == 'bolometric':
        #fixes6 = [844, 881, 925, 950]
        days6 = [1, 1.14, 1.3, 1.4] #t/t_fb
        bolom = []
        
        bolom.append(L844)
        bolom.append(L881)
        bolom.append(L925)
        bolom.append(L950)

        with open('L6.txt', 'a') as f:
            f.write(' '.join(map(str, days6))+'\n')
            f.write(' '.join(map(str, bolom)) + '\n')
            f.close()

        plt.plot(days6,bolom, '-o')
        plt.ylabel(r'$log_{10}$ Luminosity [erg/s]')
        plt.xlabel(r'$t/t_{fb}$')
        #plt.ylim(1e42,2e45)
        plt.yscale('log')

        
    if plot == 'spectra':
        plt.plot(n_logspace, L_tilda_n844, c = 'tomato', label = 't/$t_{fb}$ = 1')
        plt.plot(n_logspace, L_tilda_n881, c = 'orange', label = 't/$t_{fb}$ = 1.14')
        plt.plot(n_logspace, L_tilda_n925, c = 'yellowgreen', label = 't/$t_{fb}$ = 1.3')
        plt.plot(n_logspace, L_tilda_n950, c = 'teal', label = 't/$t_{fb}$ = 1.4')
        plt.plot(n_logspace, n3)
        plt.plot(n_logspace, n4)


        plt.xlim(12,19)
        plt.ylim(1e35, 1e45)
        plt.xlabel(r'$log\nu$ [Hz]')
        plt.yscale('log')
        plt.ylabel(r'$log\tilde{L}_\nu$ [erg/s]')

if m == 4:
    #L_tilda_n177 = L_tilda_n[4]
    L_tilda_n233 = L_tilda_n[1]
    L_tilda_n254 = L_tilda_n[2]
    L_tilda_n263 = L_tilda_n[3]

    L233 = np.trapz(L_tilda_n233, n_logspace)
    L254 = np.trapz(L_tilda_n254, n_logspace)
    L263 = np.trapz(L_tilda_n263, n_logspace)
    #L177 = np.trapz(L_tilda_n177, n_logspace)

    if plot == 'bolometric':
        #fixes6 = [233, 254, 263]
        days4 = [1, 1.2, 1.3] #t/t_fb
        bolom = []
        
        bolom.append(L233)
        bolom.append(L254)
        bolom.append(L263)
        # bolom.append(L950)

        with open('L4.txt', 'a') as f:
            f.write(' '.join(map(str, days4))+'\n')
            f.write(' '.join(map(str, bolom)) + '\n')
            f.close()
            
        plt.plot(days4,bolom, '-o')
        plt.ylabel(r'$log_{10}$ Luminosity [erg/s]')
        plt.xlabel(r'$t/t_{fb}$')
        plt.ylim(1e41,1e42)
        plt.yscale('log')

        
    if plot == 'spectra':

        plt.plot(n_logspace, L_tilda_n233, c = 'tomato', label = 't/$t_{fb}$ = 1')
        plt.plot(n_logspace, L_tilda_n254, c = 'orange', label = 't/$t_{fb}$ = 1.2')
        plt.plot(n_logspace, L_tilda_n263, c = 'yellowgreen', label = 't/$t_{fb}$ = 1.3')
        #plt.plot(n_logspace, L_tilda_n263, c = 'teal', label = 't/$t_{fb}$ = 1.4')
        plt.xlabel(r'$log\nu$ [Hz]')
        plt.yscale('log')
        plt.ylabel(r'$log\tilde{L}_\nu$ [erg/s]')
        plt.xlim(12,18)
        plt.ylim(1e32, 1e42)


plt.grid()
plt.legend()
# plt.axvline(14, color = 'tab:orange')
# plt.axvline(17, color = 'tab:orange')
# plt.axvspan(14, 17, alpha=0.3, color = 'tab:orange', label = 'UV band')
# plt.savefig('TOT_Ltilda_m' + str(m) + '.png' )
plt.show()

