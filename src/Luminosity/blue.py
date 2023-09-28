#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Sep 21 

@author: paola 

Calculate the blue (BB) curve.

NOTES FOR OTHERS:
- make changes in VARIABLES: m (power index of the BB mass), fixes (number of snapshots) and thus days
"""

# Vanilla imports
import numpy as np
import matplotlib.pyplot as plt

###
##
# VARIABLES
##
###

m = 4 # Choose BH
if m == 4:
    fixes = np.arange(233,263 + 1)
    days = [1.015, 1.025, 1.0325, 1.0435, 1.0525, 1.06, 1.07, 1.08, 1.0875, 1.0975, 1.1075, 1.115, 1.125, 1.135, 1.1425, 1.1525, 1.1625, 1.17, 1.18, 1.19, 1.1975, 1.2075, 1.2175, 1.2275, 1.235, 1.245, 1.255, 1.2625, 1.2725, 1.2825, 1.29] #t/t_fb
    # days = [4.06,4.1,4.13,4.17,4.21,4.24,4.28,4.32,4.35,4.39,4.43,4.46,4.5,4.54,4.57,4.61,4.65,4.68,4.72,4.76,4.79,4.83,4.87,4.91,4.94,4.98,5.02,5.05,5.09,5.13,5.16] #days
if m == 6:
    fixes = [844, 881, 925, 950]
    days = [1.00325, 1.13975, 1.302, 1.39425] #t/t_fb
    # days = [40, 45, 52, 55] #days