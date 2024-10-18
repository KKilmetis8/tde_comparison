#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 9

@author: konstantinos
"""

import numpy as np
pre = '/home/kilmetisk/data1/TDE/'

Mbh = '1e+06' # 10000, 100000, 1e+06
suffix = 'beta1S60n1.5Compton'
res = 'h'  # 'f' for fiducial, 'h' for HiRes, 's' for super high res,
           # 'd' double rad
if Mbh == '1e+06':
    runnos = np.arange(0, 9) + 1
    firsts = np.arange(88, 338, step = 30)
    lasts =  np.arange(98, 348, step = 30)
if Mbh == '10_000' and res == 'h':
    print('notreadyyet')
if Mbh == '100000':
    firsts = np.arange(135, 355, step = 30)    
    lasts = np.arange(145, 365, step = 30)
    runnos = np.arange(0, 8) + 1
if Mbh == '1e+06':
    firsts = np.arange(194, 404, step = 30)
    lasts = np.arange(204, 414, step = 30)
    runnos = np.arange(0, 7) + 1

for runno, first, last in zip(runnos, firsts, lasts):
    # Open file and read it
    f = open(f'{pre}slurms/proto_dist_measure.slurm', 'r')
    g = f.read()
    f.close()

    # Replace
    m = int(np.log10(float(Mbh)))
    g = g.replace('<m>', str(m))
    g = g.replace('<num>', str(runno))
    g = g.replace('<bh>', Mbh)
    g = g.replace('<first>', str(first))
    g = g.replace('<last>', str(last))
    g = g.replace('<suffix>', str(suffix))
    # Write
    name = f'{pre}slurms/forged/distance_measure_{m}_{runno}.slurm'
    h = open(name, 'w')
    h.write(g)
    h.close()

    