#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 9

@author: konstantinos
"""
import numpy as np
pre = '/home/kilmetisk/data1/TDE/'

Mbh = '10000' # 10000, 100000, 1e+06
suffix = 'beta1S60n1.5Compton'
res = 'f'  # 'f' for fiducial, 'h' for HiRes, 's' for super high res,
           # 'd' double rad
if Mbh == '10000':
    start = 90
    end = 348
    jobs = end-start
    edges = np.linspace(start, end, jobs + 1, dtype=int)
    firsts = edges[:-1]
    lasts = edges[1:] - 1  # Ensure last value is included in interval
    lasts[-1] = end  # Make sure the last interval ends at 'end'
    runnos = np.arange(0, len(firsts)) + 1
if Mbh == '10_000' and res == 'h':
    print('notreadyyet')
if Mbh == '100000':
    start = 135
    end = 365
    jobs = end - start
    edges = np.linspace(start, end, jobs + 1, dtype=int)
    firsts = edges[:-1]
    lasts = edges[1:] - 1  
    lasts[-1] = end  
    runnos = np.arange(0, len(firsts)) + 1
if Mbh == '1e+06':
    start = 200
    end = 444
    jobs = end-start
    edges = np.linspace(start, end, jobs + 1, dtype=int)
    firsts = edges[:-1]
    lasts = edges[1:] - 1  
    lasts[-1] = end  
    runnos = np.arange(0, len(firsts)) + 1

for runno, first, last in zip(runnos, firsts, lasts):
    # Open file and read it
    f = open(f'{pre}slurms/proto_red.slurm', 'r')
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
    g = g.replace('<res>', str(res))

    # Write
    name = f'{pre}slurms/forged/red_{m}_{first}.slurm'
    h = open(name, 'w')
    h.write(g)
    h.close()

    