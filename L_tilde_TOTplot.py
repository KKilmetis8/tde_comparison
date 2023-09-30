import numpy as np
import matplotlib.pyplot as plt

m = 6
plot = 'spectra'

def final_plot(m, pl):
    L_tilde_n = np.loadtxt('L_tilde_n_m'+ str(m) + '.txt')
    bolom = bolom = np.loadtxt('L_m'+ str(m) + '.txt')
    n_array = L_tilde_n[0]

    if pl == 'bolometric':
        if m == 6:
            #fixes6 = [844, 881, 925, 950]
            days = [1, 1.14, 1.3, 1.4] #t/t_fb
        if m == 4:
            #fixes4 = [233, 243, 254, 263]
            days = [1, 1.1, 1.2, 1.3] #t/t_fb

        plt.plot(days, bolom, '-o')
        plt.ylabel(r'$log_{10}$ Luminosity [erg/s]')
        plt.xlabel(r'$t/t_{fb}$')
        #plt.yscale('log')
        plt.grid()
        plt.savefig('BolomLtilda_m' + str(m) + '.png' )


    if plot == 'spectra':
        if m == 6:
            L_tilde_n844 = L_tilde_n[1]
            L_tilde_n881 = L_tilde_n[2]
            L_tilde_n925 = L_tilde_n[3]
            L_tilde_n950 = L_tilde_n[4]
            plt.plot(n_array, L_tilde_n844, c = 'tomato', label = 't/$t_{fb}$ = 1')
            plt.plot(n_array, L_tilde_n881, c = 'orange', label = 't/$t_{fb}$ = 1.14')
            plt.plot(n_array, L_tilde_n925, c = 'yellowgreen', label = 't/$t_{fb}$ = 1.3')
            plt.plot(n_array, L_tilde_n950, c = 'teal', label = 't/$t_{fb}$ = 1.4')


        if m == 4:
            L_tilde_n233 = L_tilde_n[1]
            L_tilde_n243 = L_tilde_n[3]
            L_tilde_n254 = L_tilde_n[3]
            L_tilde_n263 = L_tilde_n[3]

            times = 0
            for i in range(len(L_tilde_n233)):
                if L_tilde_n233[i]>L_tilde_n263[i]:
                    print('the first is bigger ')
                    times +=1
            print(times, 'times')

            start = 6500
            stop = 9999
            plt.plot(10**n_array, L_tilde_n233, c = 'tomato', label = 't/$t_{fb}$ = 1')
            # #plt.plot(10**n_array, L_tilde_n254, c = 'orange', label = 't/$t_{fb}$ = 1.2')
            plt.plot(10**n_array, L_tilde_n263, c = 'yellowgreen', label = 't/$t_{fb}$ = 1.3')
            #plt.plot(10**n_array[start:stop], L_tilde_n233[start:stop], c = 'tomato', label = 't/$t_{fb}$ = 1')
            #plt.plot(10**n_array[start:stop], L_tilde_n263[start:stop], c = 'yellowgreen', label = 't/$t_{fb}$ = 1.3')
            
            firstint233 = np.trapz(L_tilde_n233[0:start], n_array[0:start])
            firstint263 = np.trapz(L_tilde_n263[0:start], n_array[0:start])
            print('first 233: ', firstint233)
            print('first 263: ', firstint263)

            secondint233 = np.trapz(L_tilde_n233[start:stop], n_array[start:stop])
            secondint263 = np.trapz(L_tilde_n263[start:stop], n_array[start:stop])
            print('second 233: ', secondint233)
            print('second 263: ', secondint263)
            tot33 = firstint233 + secondint233
            tot263 = firstint263 + secondint263
            print('233: ', tot33)
            print('263: ', tot263)

        
        plt.xlabel(r'$log\nu$ [Hz]')
        plt.loglog()
        plt.ylabel(r'$log\tilde{L}_\nu$ [erg/s]')
        plt.grid()
        plt.legend()
        #plt.savefig('Ltilda_n_m' + str(m) + '.png' )


    plt.show()

