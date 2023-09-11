#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 15:48:59 2023

@author: konstantinos
"""
# Imports
import random
import os
from playsound import playsound

def finished():
    # Hold the old working dir
    old_dir = os.getcwd()
    
    # Navigate to the new one
    os.chdir('/home/konstantinos') # allazeis se panos
    
    # Choose randomly one of the files to play
    rng = random.randint(1,3)
    rng = str(rng)
    
    # Play the file
    playsound('Music/end/end' + str(rng) + '.ogg')
    
    # Return to the old working dir
    os.chdir(old_dir)
