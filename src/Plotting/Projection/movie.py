
"""
Created on Sun Dec 17

@author: paola
"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

import subprocess
import glob
import os

# Choose simlation
m = 4
check = 'fid'

path = f'Figs/denproj/{m}/denproj{m}-{check}'
output_path = f'Figs/denproj/movie{m}.mp4'

ffmpeg_command = f'ffmpeg -y -start_number 210 -i {path}%d.png -c:v libx264 -pix_fmt yuv420p -r 30 {output_path}'
subprocess.run(ffmpeg_command, shell=True)

