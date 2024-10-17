#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 16:23:50 2024

@author: konstantinos
"""

import colorsys

# Function to generate a rainbow color palette
def generate_rainbow_palette(num_colors):
    rainbow_colors = []
    for i in range(num_colors):
        hue = i / num_colors  # Spread colors evenly across the hue spectrum
        rgb = colorsys.hsv_to_rgb(hue, 1, 1)  # Full saturation and brightness
        hex_color = "#{:02x}{:02x}{:02x}".format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
        rainbow_colors.append(hex_color)
    return rainbow_colors

# Generate a rainbow color palette with 192 colors
rainbow_palette = generate_rainbow_palette(192)
print(rainbow_palette)