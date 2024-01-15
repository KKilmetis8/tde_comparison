#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jnauary 2024

@author: paola 

Calculate the old absorption opacity from Elad's data.

"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

from src.Utilities.isalice import isalice
alice, plot = isalice()

# Vanilla imports
import numpy as np

c = 3e10 #[cm/s]
alpha = 7.5657e-15 # radiation density [erg/cm^3K^4]

def opacity(Temperature, Lambdacool):
    """ Takes T and Lambda in cgs (NOT LOG) and gives you sigma_abs from ELad's code.
    Gives sigma_abs i"""
    Temperature = float(Temperature)
    Lambdacool = float(Lambdacool)
    sigma_abs = Lambdacool / (c * alpha * Temperature**4)
    return sigma_abs

if __name__ == "__main__":
    Tcool_log = np.loadtxt('src/Opacity/Tcool.txt')
    Lambdacool_log = np.loadtxt('src/Opacity/Lambdacool.txt')

    Tcool = np.power(10, Tcool_log)
    Lambdacool = np.power(10, Lambdacool_log)

    # Extend Tcool as Elad
    Tcool_pre_ext = np.logspace(2.5, np.log10(np.min(Tcool) * 0.999), 20)
    Tcool_post_ext = np.logspace(np.log10(np.max(Tcool) * 1.001), 13, 20)
    Tcool = np.concatenate([Tcool_pre_ext, Tcool, Tcool_post_ext])

    # Extend Lambda as Elad
    Lambdacool_pre_ext = Lambdacool[0] * ((Tcool[0:20] / Tcool[20])**7)
    Lambdacool_post_ext = Lambdacool[-1] * ((Tcool[-20:] / Tcool[-21])**0.5)
    Lambdacool = np.concatenate([Lambdacool_pre_ext, Lambdacool, Lambdacool_post_ext])
    
    print(len(Tcool))
    print(len(Lambdacool))

    sigma_abs = np.zeros(len(Tcool))
    for i in range(len(Tcool)):
        sigma_abs[i] = opacity(Tcool[i], Lambdacool[i])

    np.savetxt('src/Opacity/sigma_abs.txt', sigma_abs)
    np.savetxt('src/Opacity/Tcool_ext.txt', Tcool)
