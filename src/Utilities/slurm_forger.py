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
    runnos = np.arange(0, 22) + 1
    
    firsts = [80, 90, 100, 121, 142, 147, 153, 159, 163, 169, 175, 180,
              184, 205, 226, 247, 268, 288, 304, 309, 323, 330, 343]
    lasts =  [90, 100, 120, 141, 146, 152, 158, 162, 168, 174, 179, 183,
              204, 225, 246, 267, 287, 303, 308, 322, 343, 348]
if Mbh == '10_000' and res == 'h':
    first = 80
    last = 232
    firsts = np.arange(80, 232, step=4)    
    lasts = np.arange(83, 232, step=4)
    lasts[-1] += 1 # turn the last one from 231 to 232
    runnos = np.arange(0, 38) + 1

if Mbh == '100000':
    runnos = np.arange(0, 17) + 1
    firsts = [132, 140, 151, 162, 173, 184, 195, 200, 221, 242, 263,
              284, 305, 326, 344, 347, 363]
    lasts = [140, 150, 161, 172, 183, 194, 199, 220, 241, 262, 283, 
             304, 325, 343, 346, 362, 365]
if Mbh == '1e+06':
    firsts = [180, 186, 192, 196, 200, 211, 215, 219, 222, 225, 227, 229, 
              233, 234, 239, 244, 246, 249, 252, 255, 257, 260, 263, 266, 
              269, 272, 274, 277, 281, 284, 287, 294, 297, 299, 310, 321,
              332, 334, 339, 343, 354, 366, 371, 377, 383, 389, 395, 401,
              404, 408, 411]
    lasts =  [185, 191, 195, 199, 210, 214, 218, 221, 224, 226, 228, 232,
              233, 238, 243, 245, 248, 251, 254, 256, 259, 262, 265, 268, 
              271, 273, 276, 280, 283, 286, 288, 296, 298, 309, 320, 331, 
              333, 338, 342, 353, 365, 370, 376, 382, 388, 394, 400, 403, 
              407, 410, 413]
    runnos = np.arange(0,len(firsts)) + 1


for runno, first, last in zip(runnos, firsts, lasts):
    # Open file and read it
    f = open(f'{pre}slurms/proto_eladython.slurm', 'r')
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
    name = f'{pre}slurms/forged/Eladython_{m}_{res}_{runno}.slurm'
    h = open(name, 'w')
    h.write(g)
    h.close()

    