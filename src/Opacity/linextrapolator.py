#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:12:06 2024

@author: konstantinos
"""

import numpy as np
import matplotlib.pyplot as plt
import colorcet

def linearpad(D0,z0):
    factor = 100
    dz = z0[-1] - z0[-2]
    # print(np.shape(D0))
    dD = D0[:,-1] - D0[:,-2]
    
    z = [zi for zi in z0]
    z.append(z[-1] + factor*dz)
    
    z = np.array(z)
    
    #D = [di for di in D0]

    to_stack = np.add(D0[:,-1], factor*dD)
    to_stack = np.reshape(to_stack, (len(to_stack),1) )
    D = np.hstack((D0, to_stack))
    #D.append(to_stack)
    return np.array(D), z

def pad_interp(x,y,V):
    Vn, xn = linearpad(V, x)
    Vn, xn = linearpad(np.fliplr(Vn), np.flip(xn))
    Vn = Vn.T
    Vn, yn = linearpad(Vn, y)
    Vn, yn = linearpad(np.fliplr(Vn), np.flip(yn))
    Vn = Vn.T
    return xn, yn, Vn

def rich_grabapoint(x, V, slope_length = 2 ,extrarows = 3):
    '''Assumes x is sorted and lineral spaced '''
    # Low extrapolation
    xspacing_low = x[1] - x[0]
    x_extra_low = [x[0] - xspacing_low * (i + 1) for i in range(extrarows)]
    Vslope_low = (V[slope_length, :] - V[0, :]) / (x[slope_length] - x[0])
    # y = y0 + m(x-x0)
    Vextra_low = [V[0, :] + Vslope_low * (x_extra_low[i] - x[0]) for i in range(extrarows)]

    # High extrapolation
    # Low extrapolation
    xspacing_high = (x[-1] - x[-2])
    x_extra_high = [x[-1] + xspacing_high * (i + 1) for i in range(extrarows)]
    Vslope_high = (V[-1, :] - V[-slope_length, :]) / (x[-1] - x[-slope_length])
    Vextra_high = [V[-1, :] + Vslope_high * (x_extra_high[i] - x[-1]) for i in range(extrarows)]
    
    # Stack, reverse low to stack properlx
    xn = np.concatenate([x_extra_low[::-1], x, x_extra_high])
    Vn = np.vstack([Vextra_low[::-1], V, Vextra_high]) 
    
    return xn, Vn

def line(x, m, c):
    return m * x + c
from scipy.optimize import curve_fit

def fitline(x, V, slope_length = 2 ,extrarows = 3):
    # Low extrapolation
    xslope_low = (x[slope_length - 1] - x[0]) / slope_length
    x_extra_low = [x[0] - xslope_low * (i + 1) for i in range(extrarows)]
    
    # High extrapolation
    xslope_high = (x[-1] - x[-slope_length]) / slope_length
    x_extra_high = [x[-1] + xslope_high * (i + 1) for i in range(extrarows)]
    
    # Stack, reverse low to stack properlx
    xn = np.concatenate([x_extra_low[::-1], x, x_extra_high])
    
    # 2D low
    Vextra_low = []
    for i in range(V.shape[1]):  # Loop over columns of V
        params_low, _ = curve_fit(line, x[0:slope_length], V[0:slope_length, i])
        Vslope_low, Vintercept_low = params_low
        Vextra_low.append([line(x_extra_low[j], Vslope_low, Vintercept_low) for j in range(extrarows)])

    # 2D high
    Vextra_high = []
    for i in range(V.shape[1]):  # Loop over columns of V
        params_high, _ = curve_fit(line, x[-slope_length:], V[-slope_length:, i])
        Vslope_high, Vintercept_high = params_high
        Vextra_high.append([line(x_extra_high[j], Vslope_high, Vintercept_high) for j in range(extrarows)])

    # Convert back to arraxs
    Vextra_low = np.array(Vextra_low).T  # Transpose back to match dimensions
    Vextra_high = np.array(Vextra_high).T

    Vn = np.vstack([Vextra_low[::-1], V, Vextra_high]) 
    
    return xn, Vn

def extrapolator_flipper(x ,y, V, slope_length = 26, extrarows = 25,
                         what = 'rich'):
    if what == 'rich':
        xn, Vn = rich_grabapoint(x, V, slope_length, extrarows)
        yn, Vn = rich_grabapoint(y, Vn.T, slope_length, extrarows)
    if what == 'fit':
        xn, Vn = fitline(x, V, slope_length, extrarows)
        yn, Vn = fitline(y, Vn.T, slope_length, extrarows)
    return xn, yn, Vn.T

def nouveau_rich(x, y, K, what = 'scatter', slope_length = 26, extrarowsx = 100, 
                 extrarowsy = 100, highT_slope = -3.5):
    ''' 
    what, str: either scattering or absorption
    
    should be linear in log for absorption, everywhere,
    for scattering/density should be linear, irregardless of temperature,
    +opacity should never be below thompson'''
    
    X = 0.9082339738214822 # From table prescription
    thompson = 0.2 * (1 + X)
    
    # Extend x and y, adding data equally space (this suppose x,y as array equally spaced)
    # Low extrapolation
    deltaxn_low = x[1] - x[0]
    deltayn_low = y[1] - y[0] 
    x_extra_low = [x[0] - deltaxn_low * (i + 1) for i in range(extrarowsx)]
    y_extra_low = [y[0] - deltayn_low * (i + 1) for i in range(extrarowsy)]
    
    # High extrapolation
    deltaxn_high = x[-1] - x[-2]
    deltayn_high = y[-1] - y[-2]
    x_extra_high = [x[-1] + deltaxn_high * (i + 1) for i in range(extrarowsx)]
    y_extra_high = [y[-1] + deltayn_high * (i + 1) for i in range(extrarowsy)]
    
    # Stack, reverse low to stack properly
    xn = np.concatenate([x_extra_low[::-1], x, x_extra_high])
    yn = np.concatenate([y_extra_low[::-1], y, y_extra_high])
    
    Kn = np.zeros((len(xn), len(yn)))
    for ix, xsel in enumerate(xn):
        for iy, ysel in enumerate(yn):
            
            # Too cold
            if xsel < x[0]:
                deltax = x[slope_length - 1] - x[0]
                if ysel < y[0]: # Too rarefied
                    # slope_length = 2
                    deltay = y[slope_length - 1] - y[0]
                    Kxslope = (K[slope_length - 1, 0] - K[0, 0]) / deltax
                    Kyslope = (K[0, slope_length - 1] - K[0, 0]) / deltay
                    Kn[ix][iy] = K[0, 0] + Kxslope * (xsel - x[0]) + Kyslope * (ysel - y[0])
                    # if what == 'abs':
                    #    Kn[ix][iy] = K[0, 0] + Kxslope * (x[0] - xsel) + Kyslope * (y[0]-ysel)
                elif ysel > y[-1]: # Too dense
                    deltay = y[-1] - y[-slope_length] 
                    Kxslope = (K[slope_length - 1, -1] - K[0, -1]) / deltax
                    Kyslope = (K[0, -1] - K[0, -slope_length]) / deltay
                    Kn[ix][iy] = K[0, -1] + Kxslope * (xsel - x[0]) + Kyslope * (ysel - y[-1]) 
                else: # Density is inside the table
                    iy_inK = np.argmin(np.abs(y - ysel))
                    Kxslope = (K[slope_length - 1, iy_inK] - K[0, iy_inK]) / deltax
                    Kn[ix][iy] = K[0, iy_inK] + Kxslope * (xsel - x[0])
                #continue
            
            # Too hot
            elif xsel > x[-1]: 
                if ysel < y[0]: # Too rarefied
                    deltay = y[slope_length - 1] - y[0]
                    Kxslope = highT_slope #(K[-1, 0] - K[-slope_length, 0]) / deltax
                    Kyslope = (K[-1, slope_length - 1] - K[-1, 0]) / deltay
                    Kn[ix][iy] = K[-1, 0] + Kxslope * (xsel - x[-1]) + Kyslope * (ysel - y[0])        
                elif ysel > y[-1]: # Too dense
                    deltay = y[-1] - y[-slope_length] 
                    Kxslope = highT_slope #(K[-1, -1] - K[-slope_length, -1]) / deltax
                    Kyslope = (K[-1, -1] - K[-1, -slope_length]) / deltay
                    Kn[ix][iy] = K[-1, -1] + Kxslope * (xsel - x[-1]) + Kyslope * (ysel - y[-1])
                else: # Density is inside the table
                    iy_inK = np.argmin(np.abs(y - ysel))
                    Kxslope = highT_slope #(K[-1, iy_inK] - K[-slope_length, iy_inK]) / deltax
                    Kn[ix][iy] = K[-1, iy_inK] + Kxslope * (xsel - x[-1])
                
                if what == 'scattering':
                    thompson_this_den = np.log(thompson * np.exp(ysel)) # 1/cm
                    if Kn[ix][iy] < thompson_this_den:
                        # print(Kn[ix][iy], thompson_this_den)
                        Kn[ix][iy] = thompson_this_den
                #continue
            else: 
                ix_inK = np.argmin(np.abs(x - xsel))
                if ysel < y[0]: # Too rarefied, Temperature is inside table
                    # Something fucky is going on here
                    # BS change i make to avoid the line
                    # slope_length = 25
                    deltay = y[slope_length - 1] - y[0]
                    Kyslope = (K[ix_inK, slope_length - 1] - K[ix_inK, 0]) / deltay
                    Kn[ix][iy] = K[ix_inK, 0] + Kyslope * (ysel - y[0])
                    #continue
                elif ysel > y[-1]:  # Too dense, Temperature is inside table
                    deltay = y[-1] - y[-slope_length]
                    Kyslope = (K[ix_inK, -1] - K[ix_inK, -slope_length]) / deltay
                    Kn[ix][iy] = K[ix_inK, -1] + Kyslope * (ysel - y[-1])
                    #continue
                else:
                    iy_inK = np.argmin(np.abs(y - ysel))
                    Kn[ix][iy] = K[ix_inK, iy_inK]
                    # continue

    return xn, yn, Kn
if __name__ == '__main__':
    # Test data
    x = np.array([1,2,3,4,5,6,7,8,9,10])
    y = np.array([1,2,3,4,5,6,7,8,9,10])
    V = np.array([x,
                 x+1,
                 x+2,
                 x+3,
                 x+4,
                 x+5,
                 x+6,
                 x+7,
                 x+8,
                 x+9,
                 ])
    # Plot
    img = plt.pcolormesh(x, y, V, 
                         cmap = 'cet_rainbow4', vmin = -5, vmax = 15, 
                         edgecolor = 'k', lw = 0.1)
    plt.colorbar(img, )
    
    # Extrapol
    what = 'scattering'
    xn, yn, Vn = nouveau_rich(x, y, V, what)
    
    # Plot
    plt.figure()
    plt.title(what)
    img = plt.pcolormesh(xn, yn, Vn.T, 
                         cmap = 'cet_rainbow4', vmin = -5, vmax = 10, 
                         edgecolor = 'k', lw = 0.1, shading = 'gouraud')
    plt.axvline(np.min(x), c = 'white', ls = '--')
    plt.axvline(np.max(x), c = 'white', ls = '--')
    plt.axhline(np.min(y), c = 'white', ls = '--')
    plt.axhline(np.max(y), c = 'white', ls = '--')
    plt.colorbar(img, )
    plt.xlim(-1, 15)
    plt.ylim(-15, 15)


    
    