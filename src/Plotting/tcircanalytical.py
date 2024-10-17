#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 16:53:39 2024

@author: konstantinos
"""

import numpy as np
import src.Utilities.prelude as c

def tcircbytfb(mbh, mstar = 0.5, rstar = 0.47, eta = 1):
    black = c.Gcgs**(3/2) * eta / (np.pi * 2**(5/2))
    green = 1e-4 / (3.2 * c.Lsol_to_ergs)
    yellow = c.Rsol_to_cm**(-5/2) * c.Msol_to_g**(15/6)
    const = black * green * yellow
    return const * rstar**(-5/2) * mstar**(7/3) * mbh**(-5/6)

t4 =  tcircbytfb(1e4)
t5 =  tcircbytfb(1e5)
t6 =  tcircbytfb(1e6)

print('logMbh | tcirc by tfb')
print(f'4 | {t4:.2f}')
print(f'5 | {t5:.2f}')
print(f'6 | {t6:.2f}')
#%%
def Wt(Mbh, mstar = 0.5, rstar = 0.47):
    Rt = rstar * (Mbh/mstar)**(1/3)
    dE = -mstar/rstar * ((Mbh/mstar)**(1/3) + 1)
    th_entry = -rstar/Rt
    th_exit = (dE*rstar - Mbh)/(dE*rstar) * rstar/(rstar-Rt)
    I1 = th_entry**2 - th_exit**2
    I2 = th_entry - th_exit
    out = 2*dE**2*rstar**3/(Rt**2 * Mbh)
    term1 = - Rt**2/rstar**2 + Rt/rstar - 1/2
    term2 = Rt/rstar - 1
    return out*(term1*I1 + term2*I2)/dE

w4 =  Wt(1e4)
w5 =  Wt(1e5)
w6 =  Wt(1e6)

def oldtfb(Mbh, mstar = 0.5, rstar = 0.47):
    dE = -mstar/rstar * ((Mbh/mstar)**(1/3) + 1)
    a_tilde = - Mbh/(dE)
    return 2*np.pi*a_tilde**(3/2)/Mbh**(1/2)

def newtfb(Mbh, mstar = 0.5, rstar = 0.47):
    dE = -mstar/rstar * ((Mbh/mstar)**(1/3) + 1)
    w = Wt(Mbh, mstar, rstar)
    a_tilde = - Mbh/(dE+w)
    return 2*np.pi*a_tilde**(3/2)/Mbh**(1/2)

tn4 =  newtfb(1e4)
tn5 =  newtfb(1e5)
tn6 =  newtfb(1e6)

to4 =  oldtfb(1e4)
to5 =  oldtfb(1e5)
to6 =  oldtfb(1e6)
print('logMbh | Wt | new tfb | old tfb')
print(f'4 | {w4:.2f} | {tn4:.2f} | {to4:.2f}')
print(f'5 | {w5:.2f} | {tn5:.2f} | {to5:.2f}')
print(f'6 | {w6:.2f} | {tn6:.2f} | {to6:.2f}')


