#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 17:19:28 2023

@author: konstantinos
"""
import numpy as np
class borderlands:
    # Holds the corners of the points
    def __init__(self, corner_thetas, corner_phis):
        self.corner_thetas = corner_thetas
        self.corner_phis = corner_phis
    
    def __call__(self, theta, phi):
        
        # Are you over line 1
        bool1 = (phi > borderlands.oldline(theta, self.corner_phis[0:2], 
                                           self.corner_thetas[0:2]))
    
        # Are you over line 2
        bool2 = (phi > borderlands.oldline(theta, self.corner_phis[1:3], 
                              self.corner_thetas[1:3]))
        # Are you under line 3
        bool3 = (phi < borderlands.oldline(theta, self.corner_phis[2:4], 
                              self.corner_thetas[2:4])) 
        # Are you under line 4
        lastone_phi = [self.corner_phis[3], self.corner_phis[0]]
        lastone_theta = [self.corner_thetas[3], self.corner_thetas[0]]
        bool4 = (phi < borderlands.oldline(theta, lastone_phi, lastone_theta))
        
        # We want all of them to be true
        mask = bool1 * bool2 * bool3  * bool4
        return mask
    
    # This is just a function, so static.
    @staticmethod
    def line(x, phis, thetas):
        '''gives theta'''
        m = (thetas[1] - thetas[0]) / (phis[1] - phis[0]) 
        return m * (x - phis[0]) + thetas[0]
    
    @staticmethod
    def oldline(x, phis, thetas):
        '''gives phi'''
        m = (phis[1] - phis[0]) / (thetas[1] - thetas[0]) 
        return m * (x - thetas[1]) + phis[1]
        

