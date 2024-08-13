#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 18:08:56 2024

@author: konstantinos
"""
# Vanilla
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle 

# Chocolate
import src.Utilities.prelude as c
time = '0.5'
if time == '0.7':
    with open('4half164stream.pkl', 'rb') as f:
        fid_stream = pickle.load(f)
    with open('4halfHR164stream.pkl', 'rb') as f:
        HR_stream = pickle.load(f)
    with open('4halfSHR169stream.pkl', 'rb') as f:
        SHR_stream = pickle.load(f)
    with open('4half164denmax.pkl', 'rb') as f:
        fid_denmax = pickle.load(f)
    with open('4halfHR164denmax.pkl', 'rb') as f:
        HR_denmax = pickle.load(f)
    with open('4halfSHR169denmax.pkl', 'rb') as f:
        SHR_denmax = pickle.load(f)    
        
    streams = (fid_stream, HR_stream, SHR_stream)
    denmax = (fid_denmax, HR_denmax, SHR_denmax)
    del fid_stream, HR_stream, SHR_stream, fid_denmax, HR_denmax, SHR_denmax

if time == '0.5':
    with open('4half199stream.pkl', 'rb') as f:
        fid_stream = pickle.load(f)
    with open('4halfHR199stream.pkl', 'rb') as f:
        HR_stream = pickle.load(f)
    with open('4half199denmax.pkl', 'rb') as f:
        fid_denmax = pickle.load(f)
    with open('4halfHR199denmax.pkl', 'rb') as f:
        HR_denmax = pickle.load(f)
        
    streams = (fid_stream, HR_stream, )
    denmax = (fid_denmax, HR_denmax, )
    del fid_stream, HR_stream, fid_denmax, HR_denmax, 
#%% Plot stream sections
thetas = -np.linspace(-np.pi, np.pi, 100)
# fig, ax = plt.subplots(3,1, figsize = (3,6), sharex = True)
fig = plt.figure( figsize=(6,6), tight_layout = True)
ax1 = plt.subplot2grid((2,2), (0, 0), colspan = 1)
ax2 = plt.subplot2grid((2,2), (0, 1), colspan = 1)
ax3 = plt.subplot2grid((2,2), (1, 0), colspan = 2)
for i, stream, density_maxima in zip(range((len(streams))), streams, denmax):
    if i == 0:
        color = 'k'
    if i == 1:
        color = 'r'
    if i == 2:
        color = 'b'
    for check in tqdm( range(5, 90)):#ray_no) ):    
        if len(density_maxima[check]) < 1 or len(stream[check]) < 1:
            continue
        
        to_plot = np.array(stream[check]).T
        center = density_maxima[check][0]
        center_n = center[3]
        center_z = center[2]
    
        ns = to_plot[2]# - center_n
        zs = to_plot[1]# - center_z
        
        height = np.abs(np.max(zs) - np.min(zs))
        width = np.abs(np.max(ns) - np.min(ns))
        num = len(zs)
        
        ax1.scatter(thetas[check], height, c=color, marker='h')
        ax2.scatter(thetas[check], width, c=color, marker='h')
        ax3.scatter(thetas[check], num, c=color, marker='h')

ax1.set_ylim(0,3)
ax2.set_ylim(0,10)
ax1.set_xlabel('Theta')
ax2.set_xlabel('Theta')
ax3.set_xlabel('Theta')

ax3.axvline(0, color = c.AEK, lw=3)
ax2.axvline(0, color = c.AEK, lw=3)
ax1.axvline(0, color = c.AEK, lw=3)
ax1.set_ylabel('Stream Height $[R_\odot]$', fontsize = 14)
ax2.set_ylabel('Stream Width $[R_\odot]$', fontsize = 14)
ax3.set_ylabel('Cell numbers', fontsize = 14)    
ax3.set_yscale('log')
