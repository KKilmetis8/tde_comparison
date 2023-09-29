import numpy as np
import matplotlib.pyplot as plt

m = 4
plot = 'spectra'
n_min = 1e12 
n_max = 1e18
n_spacing = 10000

if plot == 'bolometric' or plot == 'spectra': 
    L_tilda_n = np.loadtxt('L_tilde_n_m'+ str(m) + '.txt')
    bolom = bolom = np.loadtxt('L_m'+ str(m) + '.txt')
    n_array = L_tilda_n[0]


    if plot == 'bolometric':
        if m == 6:
            #fixes6 = [844, 881, 925, 950]
            days = [1, 1.14, 1.3, 1.4] #t/t_fb
        if m == 4:
            #fixes4 = [177, 233,]
            days = [1, 1.2, 1.3] #t/t_fb

        plt.plot(days, bolom, '-o')
        plt.ylabel(r'$log_{10}$ Luminosity [erg/s]')
        plt.xlabel(r'$t/t_{fb}$')
        #plt.ylim(1e42,2e45)
        plt.yscale('log')
        plt.grid()
        plt.savefig('BolomLtilda_m' + str(m) + '.png' )


    if plot == 'spectra':
        if m == 6:
            L_tilda_n844 = L_tilda_n[1]
            L_tilda_n881 = L_tilda_n[2]
            L_tilda_n925 = L_tilda_n[3]
            L_tilda_n950 = L_tilda_n[4]
            plt.plot(n_array, L_tilda_n844, c = 'tomato', label = 't/$t_{fb}$ = 1')
            plt.plot(n_array, L_tilda_n881, c = 'orange', label = 't/$t_{fb}$ = 1.14')
            plt.plot(n_array, L_tilda_n925, c = 'yellowgreen', label = 't/$t_{fb}$ = 1.3')
            plt.plot(n_array, L_tilda_n950, c = 'teal', label = 't/$t_{fb}$ = 1.4')
            # plt.plot(n_array, n3)
            # plt.plot(n_array, n4)
            #plt.ylim(1e35, 1e45)


        if m == 4:
            L_tilda_n233 = L_tilda_n[1]
            #L_tilda_n254 = L_tilda_n[2]
            L_tilda_n263 = L_tilda_n[2]

            times = 0
            for i in range(len(L_tilda_n233)):
                if L_tilda_n233[i]>L_tilda_n263[i]:
                    print('the first is bigger ')
                    times +=1
            print(times, 'times')

            start = 6500
            stop = 9999
            #plt.plot(10**n_array, L_tilda_n233, c = 'tomato', label = 't/$t_{fb}$ = 1')
            # #plt.plot(10**n_array, L_tilda_n254, c = 'orange', label = 't/$t_{fb}$ = 1.2')
            #plt.plot(10**n_array, L_tilda_n263, c = 'yellowgreen', label = 't/$t_{fb}$ = 1.3')
            plt.plot(10**n_array[start:stop], L_tilda_n233[start:stop], c = 'tomato', label = 't/$t_{fb}$ = 1')
            plt.plot(10**n_array[start:stop], L_tilda_n263[start:stop], c = 'yellowgreen', label = 't/$t_{fb}$ = 1.3')

            int233 = np.trapz(L_tilda_n233[start:stop], n_array[start:stop])
            int263 = np.trapz(L_tilda_n263[start:stop], n_array[start:stop])
            print('233: ', int233)
            print('263: ', int263)

        
        plt.xlabel(r'$log\nu$ [Hz]')
        plt.loglog()
        plt.ylabel(r'$log\tilde{L}_\nu$ [erg/s]')
        plt.grid()
        plt.legend()
        #plt.savefig('Ltilda_n_m' + str(m) + '.png' )


    # plt.axvline(14, color = 'tab:orange')
    # plt.axvline(17, color = 'tab:orange')
    # plt.axvspan(14, 17, alpha=0.3, color = 'tab:orange', label = 'UV band')
    plt.show()

