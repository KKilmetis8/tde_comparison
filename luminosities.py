#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 16:05:45 2024

@author: konstantinos
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 20:49:45 2024

@author: konstantinos
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mesa_reader as mr
import os
import src.prelude as c
from src.reynolds import profile_sorter, rey_mag

class apothicarios:
    def __init__(self, name):
        self.name = name
        self.age = []
        self.lum_mesa = []
        self.lum_stefan = [] # comparing MESA logT with photosphereL
        self.lum_dyn = []
        self.lum_rcb = []
      
    def __call__(self, age_h, lum_mesa_h, lum_stefan_h, lum_dyn_h, lum_rcb_h):
        self.age.append(age_h)
        self.lum_mesa.append(lum_mesa_h)
        self.lum_stefan.append(lum_stefan_h)
        self.lum_dyn.append(lum_dyn_h)
        self.lum_rcb.append(lum_rcb_h)
      
def doer(names):
    # Count and generate profile lists
    apothikh = []
    for name in names:
        # History data wrangling
        # h_path = 'data/' + name + '/history_7.data'
        # h = mr.MesaData(h_path)
        # print(dir(h))
        # h_age = np.round(10**h.log_star_age /  1e9, 2) # Gyr
        
        # Profile data wrangling and bookeeping
        hold = apothicarios(name)
        p_path = 'data/' + name
        profiles = os.popen('ls ' + p_path + '/profile*.data').read()
        profiles = list(profiles.split("\n"))
        profiles.pop() # Remove last
        profiles = profile_sorter(profiles) # Guess what that does
        
        for profile, i in zip(profiles, range(len(profiles))):
            # Load data
            p = mr.MesaData(profile)
            r = np.power(10, p.logR)
            age = np.round(p.star_age /  1e6, 2)

            # The Phantom Luminosity. MESA
            L_mesa = p.photosphere_L
            #L_mesa /= c.Lsol
            
            # The Luminosity Wars. BB on surface
            r_cgs = r[0] * c.Rsol
            L_stefan = 4 * np.pi * r_cgs**2 * c.sigma* np.power(10, p.logT[0])**4
            L_stefan /= c.Lsol
            
            # Revenge of the Luminosity. BB on dynamo surface
            _, reynolds_mag_number, _= rey_mag(p)
            R_dynamo_active = r[reynolds_mag_number > c.critical_rey_mag_num]
            Rdyn = R_dynamo_active[0]
            idx = np.argmin(np.abs(r - Rdyn)) 
            Rdyn *= c.Rsol # [cgs]
            L_dyn = 4 * np.pi * Rdyn**2 * c.sigma* np.power(10, p.logT[idx])**4
            L_dyn /= c.Lsol

            # A New Luminosity. BB on Radiative Convective Boundary
            rad = p.gradr
            conv = p.grada
            cmorethanb = r[conv < rad]
            RCB = cmorethanb[0]
            idx_rcb = np.argmin(np.abs(r - RCB))
            RCB *= c.Rsol
            L_rcb = 4 * np.pi * RCB**2 * c.sigma* np.power(10, p.logT[idx_rcb])**4
            L_rcb /= c.Lsol
            
            # Save
            hold(age, L_mesa, L_stefan, L_dyn, L_rcb)
        apothikh.append(hold)
    return apothikh


def plotter(names):
   
    colors = [c.darkb, c.cyan, c.yellow, c.reddish]
    
    # Makes the calculations
    planets = doer(names) 
    
    fig, axs = plt.subplots(1,1, tight_layout = True, sharex = True,
                           figsize = (4,4))
    
    for planet in planets:
        axs.plot(planet.age, planet.lum_mesa, color = colors[0], 
                  label = 'MESA')
        axs.plot(planet.age, planet.lum_stefan, color = colors[1], 
                 linestyle = '--', label = 'BB Surface')
        axs.plot(planet.age, planet.lum_dyn, color = colors[2],
                 linestyle = '-.', label = r'BB $R_{dyn}$')
        axs.plot(planet.age, planet.lum_rcb, color = colors[3],
                 linestyle = ':', label = r'BB $R_{rcb}$')

    # Make nice
    axs.set_ylabel('Luminosity [$L_\odot$]', fontsize = 14)
    axs.set_xlabel(r'Age [Myr]',fontsize = 15)
    #axs.set_ylim(-1e-5,1e-5)
    axs.set_yscale('log')
    axs.set_title('Different kinds of luminosities on hot jupiter')
    axs.legend(ncols = 2, fontsize = 8)# loc = 'lower center')

#%%    

name = 'jup17'
labels = ['Hot Jup']
plotter([name])
