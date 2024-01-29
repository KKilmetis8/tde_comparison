
"""
Created on Sun Dec 17

@author: paola
"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

import subprocess
import glob
import os

# Choose simulation
m = 6
check = 'fid'
if m == 6:
    start = 844
    digit = '04'
if m == 4:
    start = 130
    digit = 3

path = f'Figs/denproj/{m}/denproj{m}-{check}'
output_path = f'Final_plot/movie{m}.mp4'

ffmpeg_command = f'ffmpeg -y -start_number {start} -i {path}%{digit}d.png -c:v libx264 -pix_fmt yuv420p -r 100 {output_path}'
subprocess.run(ffmpeg_command, shell=True)

