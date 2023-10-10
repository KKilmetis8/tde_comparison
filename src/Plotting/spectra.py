import numpy as np
import matplotlib.pyplot as plt

# FUNCTIONS

def select_days(m):
    if m == 6:
        #fixes6 = [844, 881, 925, 950]
        days = [1, 1.14, 1.3, 1.4] #t/t_fb
    if m == 4:
        #fixes4 = [233, 243, 254, 263]
        days = [1, 1.1, 1.2, 1.3, 1.57, 1.7, 1.83] #t/t_fb
    return days

def final_plot(m, pl, xaxis_T = False, telescope = False, check = False):
    if pl == 'bolometric':
        days = select_days(m)
        bolom = bolom = np.loadtxt('L_m'+ str(m) + '.txt')
        if m == 6:
            plt.plot(days, bolom, '-o', c = 'olivedrab')
            plt.text(1, 1e44, '$M_{BH}=10^6M_{sun}$', fontsize = 18)
        if m == 4:
            plt.plot(days, bolom, '-o', c = 'slateblue')
            plt.text(1, 1e41, '$M_{BH}=10^4M_{sun}$', fontsize = 18)
        plt.yscale('log')
        plt.ylabel(r'$log_{10}$ Luminosity [erg/s]', fontsize = 18)
        plt.xlabel(r'$t/t_{fb}$', fontsize = 18)
        plt.grid()
        plt.legend()
        plt.savefig('BolomLtilda_m' + str(m) + '.png' )

    if pl == 'spectra':
        L_tilde_n = np.loadtxt('L_tilde_n_m'+ str(m) + '.txt')
        x_array = L_tilde_n[0]

        if xaxis_T == True:
            # from Wien law: n_peak = 5.879e10 Hz/K * T where n = 10^x
            x_array = 10**x_array / (5.879e10)
            plt.xlabel('$log_{10}T [K]$', fontsize = 18)
            plt.loglog()
        else: 
            plt.xlabel(r'$log_{10}\nu$ [Hz]', fontsize = 18)
            plt.yscale('log')
        
        if m == 6:
            L_tilde_n844 = L_tilde_n[1]
            L_tilde_n881 = L_tilde_n[2]
            L_tilde_n925 = L_tilde_n[3]
            L_tilde_n950 = L_tilde_n[4]
            plt.plot(x_array, L_tilde_n844, c = 'tomato', label = 't/$t_{fb}$ = 1')
            plt.plot(x_array, L_tilde_n881, c = 'orange', label = 't/$t_{fb}$ = 1.14')
            plt.plot(x_array, L_tilde_n925, c = 'yellowgreen', label = 't/$t_{fb}$ = 1.3')
            plt.plot(x_array, L_tilde_n950, c = 'teal', label = 't/$t_{fb}$ = 1.4')
            plt.text(12.5, 1e27, '$M_{BH}=10^6M_{sun}$', fontsize = 18)

        if m == 4:
            L_tilde_n233 = L_tilde_n[1]
            L_tilde_n243 = L_tilde_n[2]
            L_tilde_n254 = L_tilde_n[3]
            L_tilde_n263 = L_tilde_n[4]
            L_tilde_n293 = L_tilde_n[5]
            L_tilde_n308 = L_tilde_n[6]
            L_tilde_n322 = L_tilde_n[7]


            plt.plot(x_array, L_tilde_n233, c = 'r', label = 't/$t_{fb}$ = 1')
            plt.plot(x_array, L_tilde_n243, c = 'orange', label = 't/$t_{fb}$ = 1.1')
            plt.plot(x_array, L_tilde_n254, c = 'yellowgreen', label = 't/$t_{fb}$ = 1.2')
            plt.plot(x_array, L_tilde_n263, c = 'teal', label = 't/$t_{fb}$ = 1.3')
            plt.plot(x_array, L_tilde_n293, c = 'b', label = 't/$t_{fb}$ = 1.6')
            plt.plot(x_array, L_tilde_n308, c = 'blueviolet', label = 't/$t_{fb}$ = 1.7')
            plt.plot(x_array, L_tilde_n322, c = 'magenta', label = 't/$t_{fb}$ = 1.8')
            plt.text(12.5, 1e25, '$M_{BH}=10^4M_{sun}$', fontsize = 18)
            # plt.xlim(12,18)

            if check:
                start = 6500
                stop = 9999
                # plt.plot(x_array[start:stop], L_tilde_n233[start:stop], c = 'tomato', label = 't/$t_{fb}$ = 1')
                # plt.plot(x_array[start:stop], L_tilde_n263[start:stop], c = 'yellowgreen', label = 't/$t_{fb}$ = 1.3')
                
                firstint233 = np.trapz(L_tilde_n233[0:start], x_array[0:start])
                firstint263 = np.trapz(L_tilde_n263[0:start], x_array[0:start])
                print('first 233: ', firstint233)
                print('first 263: ', firstint263)

                secondint233 = np.trapz(L_tilde_n233[start:stop], x_array[start:stop])
                secondint263 = np.trapz(L_tilde_n263[start:stop], x_array[start:stop])
                print('second 233: ', secondint233)
                print('second 263: ', secondint263)
                tot33 = firstint233 + secondint233
                tot263 = firstint263 + secondint263
                print('233: ', tot33)
                print('263: ', tot263)

        plt.ylabel(r'$log_{10}\tilde{L}_\nu$ [erg/sHz]', fontsize = 18)
        plt.grid()

        if telescope == False:
            plt.legend()
            plt.savefig('spectra_m' + str(m) + '.png' )

            #If you want the plot without logscale
            # plt.xlim(13,17)
            # plt.ylabel(r'$\tilde{L}_\nu$ [erg/s]', fontsize = 18)
            # plt.grid()
            # plt.savefig('NOlog_spectra_m' + str(m) + '.png' )

        if telescope: 
            ultrasat_min = 1.03e15
            ultrasat_max = 1.3e15
            r_min = 4.11e14
            r_max = 5.07e14
            g_min = 5.66e14
            g_max = 7.48e14

            plt.xlim(14,16)
            plt.ylim(1e22,1e30)
            plt.axvline(np.log10(ultrasat_min), color = 'b')
            plt.axvline(np.log10(ultrasat_max), color = 'b')
            plt.axvspan(np.log10(ultrasat_min), np.log10(ultrasat_max), alpha=0.4, color = 'b')
            plt.text(np.log10(ultrasat_min)+0.04,1e23,'ULTRASAT', rotation = 90)

            plt.axvline(np.log10(r_min), color = 'r')
            plt.axvline(np.log10(r_max), color = 'r')
            plt.axvspan(np.log10(r_min), np.log10(r_max), alpha=0.4, color = 'r')
            plt.text(np.log10(r_min)+0.04,1e23,'R-band ZTF', rotation = 90)

            plt.axvline(np.log10(g_min), color = 'orange')
            plt.axvline(np.log10(g_max), color = 'orange')
            plt.axvspan(np.log10(g_min), np.log10(g_max), alpha=0.4, color = 'orange')
            plt.text(np.log10(g_min)+0.05,1e23,'G-band ZTF', rotation = 90)
            plt.legend()
            plt.savefig('telescope_spectra_m' + str(m) + '.png' )

# MAIN
m = 6

# plt.figure(figsize=(15,8))
# plotbolo = final_plot(m,'bolometric')
plt.figure(figsize=(15,8))
plotspectra = final_plot(m,'spectra', True)
plt.show()