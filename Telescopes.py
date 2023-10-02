""""
autohor: Paola

Plots in the band of ULTRASAT and ZTF.

"""
import numpy as np
import matplotlib.pyplot as plt

# FUNCTIONS

def select_days(m):
    if m == 6:
        #fixes6 = [844, 881, 925, 950]
        days = [1, 1.14, 1.3, 1.4] #t/t_fb
    if m == 4:
        #fixes4 = [233, 243, 254, 263,  293, 308, 322]
        days = [1, 1.1, 1.2, 1.3, 1.57, 1.7, 1.83] #t/t_fb
    return days

def final_plot(m, telescope):
    bolom = bolom = np.loadtxt('L_m'+ str(m) + '.txt')
    L_tilde_n = np.loadtxt('L_tilde_n_m'+ str(m) + '.txt')
    x_array = L_tilde_n[0]
    n_array = 10**x_array

    if telescope == 'ultrasat':
        n_min = 1.03e15
        n_max = 1.3e15
        
    if telescope == 'ztf':
        r_min = 4.11e14
        r_max = 5.07e14
        g_min = 5.66e14
        g_max = 7.48e14
        n_min = r_min
        n_max = g_max

        plt.axvline(r_min, color = 'bisque')
        plt.axvline(r_max, color = 'bisque')
        plt.axvspan(r_min, r_max, alpha=0.4, color = 'bisque')
        plt.text(5e14,2.1e-17,'R-band ZTF', rotation = 90)

        plt.axvline(g_min, color = 'orange')
        plt.axvline(g_max, color = 'orange')
        plt.axvspan(g_min, g_max, alpha=0.4, color = 'orange')
        plt.text(g_min+100,2.1e-17,'G-band ZTF', rotation = 90)
    
    if m == 6: 
        L_tilde_n844 = L_tilde_n[1]/bolom[0]
        L_tilde_n881 = L_tilde_n[2]/bolom[1]
        L_tilde_n925 = L_tilde_n[3]/bolom[2]
        L_tilde_n950 = L_tilde_n[4]/bolom[3]
        plt.plot(n_array, L_tilde_n844, c = 'r', label = 't/$t_{fb}$ = 1')
        plt.plot(n_array, L_tilde_n881, c = 'orange', label = 't/$t_{fb}$ = 1.14')
        plt.plot(n_array, L_tilde_n925, c = 'yellowgreen', label = 't/$t_{fb}$ = 1.3')
        plt.plot(n_array, L_tilde_n950, c = 'teal', label = 't/$t_{fb}$ = 1.4')
        plt.text(12, 1e-20, '$M_{BH}=10^6M_{sun}$', fontsize = 18)

    if m == 4:
        L_tilde_n233 = L_tilde_n[1]/bolom[0]
        L_tilde_n243 = L_tilde_n[2]/bolom[1]
        L_tilde_n254 = L_tilde_n[3]/bolom[2]
        L_tilde_n263 = L_tilde_n[4]/bolom[3]
        L_tilde_n293 = L_tilde_n[5]/bolom[4]
        L_tilde_n308 = L_tilde_n[6]/bolom[5]
        L_tilde_n322 = L_tilde_n[7]/bolom[6]

        plt.plot(n_array, L_tilde_n233, c = 'r', label = 't/$t_{fb}$ = 1')
        plt.plot(n_array, L_tilde_n243, c = 'orange', label = 't/$t_{fb}$ = 1.1')
        plt.plot(n_array, L_tilde_n254, c = 'yellowgreen', label = 't/$t_{fb}$ = 1.2')
        plt.plot(n_array, L_tilde_n263, c = 'teal', label = 't/$t_{fb}$ = 1.3')
        plt.plot(n_array, L_tilde_n293, c = 'b', label = 't/$t_{fb}$ = 1.6')
        plt.plot(n_array, L_tilde_n308, c = 'blueviolet', label = 't/$t_{fb}$ = 1.7')
        plt.plot(n_array, L_tilde_n322, c = 'magenta', label = 't/$t_{fb}$ = 1.8')
        plt.text(12, 2e-16, '$M_{BH}=10^4M_{sun}$', fontsize = 18)

    plt.xlabel(r'$log_{10}\nu$ [Hz]', fontsize = 18)
    plt.xlim(n_min,n_max)
    plt.loglog()
    plt.ylabel(r'$log_{10}(\frac{\tilde{L}_\nu}{L})$ [erg/s]', fontsize = 18)
    plt.grid()
    plt.legend()


# MAIN
m = 6

plt.figure(figsize=(15,8))
plt.ylim(2e-17,2e-16) #m=6
# plt.ylim(4.2e-16,6.8e-16) #m = 4
ultrasat = final_plot(m, 'ultrasat')
plt.text(1.05e15, 3e-17, '$M_{BH}=10^6M_{sun}$ \n ULTRASAT', fontsize = 18) # m=6
# plt.text(1.05e15, 2e-16, '$M_{BH}=10^4M_{sun}$ \n ULTRASAT', fontsize = 18) # m=4
plt.savefig('ultrasat_m'+ str(m) + '.png') 

plt.figure(figsize=(15,8))
Rztf = final_plot(m, 'ztf')
plt.ylim(2e-17,2e-16) #m=6
# plt.ylim(2e-16,5e-16) m=4
plt.text(6.5e14, 2.2e-17, '$M_{BH}=10^6M_{sun}$ \n ZTF', fontsize = 18) #m=6
# plt.text(6.5e14, 2.2e-16, '$M_{BH}=10^4M_{sun}$ \n ZTF', fontsize = 18) #m=4

plt.savefig('ztf_m'+ str(m) + '.png') 
plt.show()