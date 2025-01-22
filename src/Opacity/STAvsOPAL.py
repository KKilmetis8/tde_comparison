#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:53:55 2024

@author: konstantinos
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import colorcet as cc

import src.Utilities.prelude as c
from src.Opacity.LTE_loader import T_opac_ex, Rho_opac_ex, rossland_ex, plank_ex
opac = 'TOP'
path = f'src/Opacity/{opac}_data/'
table = 'top2'
clean = True

if clean and opac == 'opal':
    # Read the file into a list of lines
    with open(f'{path}{table}.txt', "r") as file:
        lines = file.readlines()
        
    # Initialize a list to hold the filtered lines
    filtered_lines = []
    skip_next = False
    
    # Iterate through the lines and filter
    for line in lines:
        if skip_next:
            # Skip the line immediately following "FOR LOG"
            skip_next = False
            continue
        if "FOR LOG" in line:
            # Mark to skip the next line
            skip_next = True
            continue
        filtered_lines.append(line)
    
    # Save the filtered lines back to a new file
    cleaned_file_path = f"{path}/{table}_clean.txt"
    with open(cleaned_file_path, "w") as file:
        file.writelines(filtered_lines)
    df = pd.read_csv(cleaned_file_path, sep='\s+', comment = 'L')
    
    # Get kappa
    Ts = np.unique(df['t=log(T)'])
    rhos = np.unique(df['r=log(rho)'])

    kappa = np.zeros((len(Ts), len(rhos)))
    for i, T in enumerate(Ts):
        Tmatch = np.where( np.abs(T - df['t=log(T)']) < 1e-10)
        for j, rho in enumerate(rhos):
            rhomatch = np.argmin( np.abs(rho - df['r=log(rho)'].iloc[Tmatch]))  
            idx =  df['r=log(rho)'].iloc[Tmatch].index[rhomatch]
            kappa[i,j] = df['G=log(ross)'].iloc[idx]

elif clean and opac == 'TOP':
    # Step 1: Read the raw data into a list of strings
    with open(f'{path}{table}.txt', 'r') as file:
        raw_data = file.readlines()
    
    # Step 2: Initialize variables to store the parsed data
    parsed_data = []
    current_temp = None
    current_table = []
    
    # Step 3: Loop through each line and parse the data
    for line in raw_data:
        # Strip any extra whitespace
        line = line.strip()
        
        # Check if the line is a temperature header (e.g., "T=  4.0000E-03")
        if line.startswith('Density'):
            if current_table:
                # Add the current temperature block to the parsed data
                parsed_data.append((current_temp, 
                                    pd.DataFrame(current_table,
                                                 columns=['Density', 
                                                          'Ross_Opa',
                                                          'Planck_Opa', 
                                                          'No_Free',
                                                          'Av_Sq_Free'])))
                current_table = []  # Reset for the next temperature section
    
            # Extract the temperature value and store it
            current_temp = float(line.split('=')[1].strip())
        
        # Check if the line is data and not the header
        elif len(line.split()) == 5:  # The data lines have 5 columns
            values = line.split()
            density, ross_opacity, planck_opacity, no_free, av_sq_free = map(float, values)
            current_table.append([density, ross_opacity, planck_opacity, no_free, av_sq_free])

    # Add the last section
    if current_table:
        parsed_data.append((current_temp, pd.DataFrame(current_table, columns=['Density', 'Ross_Opa', 'Planck_Opa', 'No_Free', 'Av_Sq_Free'])))
    
    # Step 4: Combine all sections into one DataFrame with a 'Temperature' column
    all_data = pd.DataFrame()
    for temp, df in parsed_data:
        df['Temperature'] = np.log10(temp * c.kEv_to_K)  # Add the temperature as a column
        all_data = pd.concat([all_data, df], ignore_index=True)
    all_data['Density'] = np.log10(all_data['Density'])
    all_data['Ross_Opa'] = np.log10(all_data['Ross_Opa'])

    # Step 6: Reshape the DataFrame into a grid format if needed
    unique_temps = np.unique(all_data['Temperature'])
    unique_densities = np.unique(all_data['Density'])
    
    # Create a 2D grid for the Rosseland opacity data
    opacity_grid = np.zeros((len(unique_densities), len(unique_temps)))
    
    for i, temp in enumerate(unique_temps):
        for j, dens in enumerate(unique_densities):
            mask = (all_data['Temperature'] == temp) & (all_data['Density'] == dens)
            if mask.any():
                opacity_grid[j, i] = all_data.loc[mask, 'Ross_Opa'].values[0]
    kappa = opacity_grid
    Ts = unique_temps
    rhos = unique_densities

opac_kind = 'LTE'
opac_path = f'src/Opacity/{opac_kind}_data'
T_cool = np.loadtxt(f'{opac_path}/T.txt')
Rho_cool = np.loadtxt(f'{opac_path}/rho.txt')
rossland = np.loadtxt(f'{opac_path}/ross.txt')

T_us = np.log10(np.exp(T_cool))
rho_us = np.log10(np.exp(Rho_cool))
ross_us = np.log10(np.exp(rossland) / np.exp(Rho_cool)) # 1/cm / g/cm3 = cm2/g

T_us_big = np.log10(np.exp(T_opac_ex))
rho_us_big = np.log10(np.exp(Rho_opac_ex))
ross_us_big = np.log10(np.exp(rossland_ex) / np.exp(Rho_opac_ex)) # 1/cm / g/cm3 = cm2/g 
#%% Plot
fig, ax = plt.subplots(1,2, sharex = True, sharey = True)
cmin = -8
cmax = 8
cmap = cc.m_CET_CBL2_r
new_cmap = cmap(np.linspace(0, 1, 256))  # 256 levels of the original colormap
from matplotlib.colors import ListedColormap
new_cmap = ListedColormap(new_cmap)

# Modify the color for a specific value
X = 0.9082339738214822  # 0.7381
specific_value = np.log10(0.2*(1+X))
# specific_color = [1.0, 0.0, 0.0, 1.0]  # Red in RGBA
# norm = plt.Normalize(vmin=cmin, vmax=cmax)
# # Find the normalized index of the specific value
# specific_index = int(norm(specific_value) * (len(new_cmap.colors) - 1))
# new_cmap.colors[specific_index] = specific_color


img = ax[0].pcolormesh(Ts, rhos, kappa, shading = 'gouraud',
                     cmap = new_cmap, vmin = cmin, vmax = cmax)
img2 = ax[1].pcolormesh(T_us, rho_us, ross_us.T,
                     cmap = new_cmap, vmin = cmin, vmax = cmax)
img2 = ax[1].pcolormesh(T_us_big, rho_us_big, ross_us_big.T,
                     cmap = new_cmap, vmin = cmin, vmax = cmax)
ax[1].axhline(np.min(rho_us), c = 'r', ls ='--')
ax[1].axhline(np.max(rho_us), c = 'r', ls ='--')
ax[1].axvline(np.min(T_us), c = 'r', ls ='--')
ax[1].axvline(np.max(T_us), c = 'r', ls ='--')

# plt.xlim(3.8, 7.42)
# plt.ylim(-15, 1)
cb = fig.colorbar(img2)
cb.set_label( '$\log(\kappa_\mathrm{Ross}$) [cm$^2$/g]')
cbax = cb.ax
# cbax.text(4, 1, '$\log(\kappa_\mathrm{Ross}$) [cm$^2$/g]', rotation = 90)
# Get existing ticks and add a new one
# existing_ticks = cb.get_ticks()
# new_tick = specific_value # 'log(0.2(1+X))'  # Example of an extra tick
# new_ticks = np.append(existing_ticks, new_tick)
# cb.set_ticks(new_ticks)

# # Format labels
# new_labels = []
# for tick in new_ticks:
#     if tick !=specific_value:
#         new_labels.append(f"{tick:.1f}")
#     else:
#         new_labels.append("log(0.2(1+X))")
# cb.ax.set_yticklabels(new_labels)

ax[0].set_xlabel('$\log (T) $ [K]')
ax[0].set_ylabel(r'$\log (\rho ) $ g/cm$^3$')
ax[0].set_title('TOP')
ax[1].set_title('STA')
ax[0].set_ylim(-15, 4)
ax[0].set_xlim(3.5, 8)
#%% Rho plot
fig, axs = plt.subplots(2,3, figsize = (7,7), 
                        tight_layout = True, sharey = True)
target_rhos = [1e-10, 1e-9, 1e-6, 1e-4, 1e-2, 1e0]
for target_rho, ax in zip(target_rhos, axs.flatten()):
    rho_idx = np.argmin( np.abs(rhos - np.log10(target_rho)))
    if opac == 'TOP':
        sliced_kappa = kappa.T[:,rho_idx]
    else:
        sliced_kappa = kappa[:,rho_idx]
    # Us
    rho_idx_us = np.argmin(np.abs(rho_us - np.log10(target_rho)))
    sliced_kappa_us = ross_us[:,rho_idx_us]
    
    ax.plot(Ts, sliced_kappa, c='k', label = opac)
    ax.plot(T_us, sliced_kappa_us, c='r', label = 'STA')
    ax.axhline(specific_value, c = 'b', ls = '--' )
    ax.set_title(f'Log Density {np.log10(target_rho)} g/cm$^3$')
    # ax.set_ylim(-2,2)
    # ax.set_xlim(1, 7)
axs[1,0].text(3.6, -0.3, 'log(0.2(1+X))', c='b', fontsize = 9)
axs[1,0].set_xlabel('$\log ( T ) $ [K]')
axs[1,0].set_ylabel('$\log( \kappa_\mathrm{Planck}$ ) [cm$^2$/g]')
plt.legend()
#%% T plot
fig, axs = plt.subplots(2,2, figsize = (5,5), 
                        tight_layout = True, sharey = True)
target_Ts = [1e4, 1e5, 1e6, 1e7]
for target_T, ax in zip(target_Ts, axs.flatten()):
    T_idx = np.argmin( np.abs(Ts - np.log10(target_T)))
    if opac == 'TOP':
        sliced_kappa = kappa.T[T_idx,:]
    else:
        sliced_kappa = kappa[T_idx,:]
    
    T_idx_us = np.argmin(np.abs( T_us - np.log10(target_T)))
    sliced_kappa_us = ross_us[T_idx_us,:]
    
    ax.plot(rhos, sliced_kappa, c='k' , label = opac)
    ax.plot(rho_us, sliced_kappa_us,c='r', label = 'STA')
    ax.set_title(f'Log T {np.log10(target_T)} K')

axs[1,0].set_xlabel(r'$\log ( \rho ) $ g/cm$^3$')
axs[1,0].set_ylabel('$\log( \kappa_\mathrm{Planck}$ ) [cm$^2$/g]')
plt.legend()