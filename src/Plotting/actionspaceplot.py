#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 12:38:03 2024

@author: konstantinos
"""

import numpy as np
import matplotlib.pyplot as plt
import colorcet
from src.Utilities.parser import parse

plt.rcParams['text.usetex'] = False

args = parse()
sim = args.name
path = f'/home/s3745597/data1/TDE/tde_comparison/data/actionspace/{sim}'
daypath = f'/home/s3745597/data1/TDE/{sim}'
fixes = np.arange(args.first, args.last + 1)
savepath = f'/home/s3745597/data1/TDE/figs/actionspace/{sim}'
for fix in fixes:
    j = np.load(f'{path}/j_{fix}.npy')
    ecc = np.load(f'{path}/ecc_{fix}.npy')
    orb = np.load(f'{path}/orb_{fix}.npy')
    day = np.loadtxt(f'{daypath}/snap_{fix}/tbytfb_{fix}.txt')
    
    # Plot it
    fig = plt.figure(figsize=(4,4), dpi=300)
    step = 10
    plt.scatter(-orb[::step], j[::step], c=ecc[::step], 
            s = 1, cmap = 'cet_rainbow4', vmin = 0.5, vmax = 1) 
    cb = plt.colorbar()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e0, 1e6)
    plt.ylim(1e2, 5e4)
    plt.xlabel('Orbital Energy')
    plt.ylabel('Angular Momentum')
    plt.text(0.55, 0.76, f'{day:.2f} tFB', fontsize = 13, 
            transform = fig.transFigure,
            bbox=dict(facecolor='whitesmoke', edgecolor='black'))
    cb.set_label('Eccentricity')

    plt.savefig(f'{savepath}/as_{fix}.png', dpi=300)
    plt.close()
