""""
autohor: Paola

Plots in the band of ULTRASAT and ZTF.

"""
import numpy as np
import matplotlib.pyplot as plt

# VARIABLES

z = 0.3 #redshift

# FUNCTIONS

def select_days(m):
    if m == 6:
        #fixes6 = [844, 881, 925, 950]
        days = [1, 1.14, 1.3, 1.4] #t/t_fb
    if m == 4:
        #fixes4 = [233, 243, 254, 263,  293, 308, 322]
        days = [1, 1.1, 1.2, 1.3, 1.57, 1.7, 1.83] #t/t_fb
    return days

def select_band(m, telescope, redshift = False):
    L_tilde_n = np.loadtxt('L_tilde_n_m'+ str(m) + '.txt')
    x_array = L_tilde_n[0]
    n_array = 10**x_array

    if telescope == 'ultrasat':
        n_min = 1.03e15
        n_max = 1.3e15
    
    if telescope == 'Gztf':
        n_min = 5.66e14
        n_max = 7.48e14

    if telescope == 'Rztf':
        n_min = 4.11e14
        n_max = 5.07e14

    if redshift:
        n_min = (1+z) * n_min
        n_max = (1+z) * n_max

    position_start = 0
    position_end = 0
    for i in range(3000,len(n_array)):
        if n_array[i]>n_min:
            position_start = i
            print('start poisition:', position_start)
            print(n_array[position_start])
            break
    for j in range(position_start,len(n_array)):
        if n_array[j]>n_max:
            position_end = j-1
            print('end poisition:', position_end)
            print(n_array[position_end])
            break

    n_array_telescope = n_array[position_start:position_end+1]
    delta_n = n_array_telescope[-1]-n_array_telescope[0]

    if m == 6: 
        L_telescope844 = (L_tilde_n[1][position_end] + L_tilde_n[1][position_start]) * delta_n /2
        L_telescope881 = (L_tilde_n[2][position_end] + L_tilde_n[2][position_start]) * delta_n /2
        L_telescope925 = (L_tilde_n[3][position_end] + L_tilde_n[3][position_start]) * delta_n /2
        L_telescope950 = (L_tilde_n[4][position_end] + L_tilde_n[4][position_start]) * delta_n /2
        L_telescope = [L_telescope844, L_telescope881, L_telescope925, L_telescope950]
        plt.text(1e14, 1e-20, '$M_{BH}=10^6M_{sun}$', fontsize = 18)


    if m == 4:
        L_telescope233 = (L_tilde_n[1][position_end] + L_tilde_n[1][position_start]) * delta_n /2
        L_telescope243 = (L_tilde_n[2][position_end] + L_tilde_n[2][position_start]) * delta_n /2
        L_telescope254 = (L_tilde_n[3][position_end] + L_tilde_n[3][position_start]) * delta_n /2
        L_telescope263 = (L_tilde_n[4][position_end] + L_tilde_n[4][position_start]) * delta_n /2
        L_telescope293 = (L_tilde_n[5][position_end] + L_tilde_n[5][position_start]) * delta_n /2
        L_telescope308 = (L_tilde_n[6][position_end] + L_tilde_n[6][position_start]) * delta_n /2
        L_telescope322 = (L_tilde_n[7][position_end] + L_tilde_n[7][position_start]) * delta_n /2
        L_telescope = [L_telescope233, L_telescope243, L_telescope254, L_telescope263, L_telescope293, L_telescope308, L_telescope322]

    return L_telescope

def final_plot(m, redshift = False):
    bolom = np.loadtxt('L_m'+ str(m) + '.txt')
    days = select_days(m)
    ultrasat = select_band(m, 'ultrasat', redshift)
    Gztf = select_band(m, 'Gztf', redshift)
    Rztf = select_band(m, 'Rztf', redshift)
    
    plt.plot(days, bolom, 'o-', c = 'forestgreen', label = 'Bolometric')
    plt.plot(days, ultrasat, 'o-', c = 'royalblue', label = 'ULTRASAT')
    plt.plot(days, Gztf, 'o-', c = 'sandybrown', label = 'ZTF G-band')
    plt.plot(days, Rztf, 'o-', c = 'coral', label = 'ZTF R-band')
    
    plt.xlabel(r'$t/t_{fb}$', fontsize = 18)
    plt.ylabel(r'$log_{10}(\nu\tilde{L}_\nu)$ [erg/s]', fontsize = 18)
    plt.yscale('log')
    plt.grid()
    plt.legend()

    if redshift:
        if m == 4:
            plt.text(1, 1e39, '$M_{BH}=10^4M_{sun}$\n' + f'redshift = {z}', fontsize = 18)
        if m == 6:
            plt.text(1.3, 1e41, '$M_{BH}=10^6M_{sun}$\n' + f'redshift = {z}', fontsize = 18)
        plt.savefig('n_Ln_band'+ str(m) + f'_z{z}.png') 
    else:
        if m == 4:
            plt.text(1, 1e39, '$M_{BH}=10^4M_{sun}$\n' + f'redshift = {z} $', fontsize = 18)
        if m == 6:
            plt.text(1.3, 1e41, '$M_{BH}=10^6M_{sun}$', fontsize = 18)
        plt.savefig('n_Ln_band'+ str(m) + '.png') 
    plt.show()


# MAIN
m = 6

plt.figure(figsize=(15,8))
final_plot(m)
