
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

# path = 'Figs/denproj/' + str(m) + '/denproj' + str(m) + '-' + check 
# # Use glob to expand the wildcard and get a list of matching files
# file_list = glob.glob(path + '.png')

# # Check if any files were found
# if not file_list:
#     print("No PNG files found in the specified directory.")
# else:
#     # Construct the FFmpeg command
#     command = [
#         'ffmpeg',
#         '-f', 'image2',
#         '-r', '1/5',
#         '-i'] + file_list + [
#         '-y',
#         'Figs/denproj/movie' + str(m) + '.mp4'
#     ]

# # Run the command
# subprocess.run(command)

# import cv2
# import os

# image_folder = 'images'
# video_name = 'video.avi'

# images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
# frame = cv2.imread(os.path.join(image_folder, images[0]))
# height, width, layers = frame.shape

# video = cv2.VideoWriter(video_name, 0, 1, (width,height))

# for image in images:
#     video.write(cv2.imread(os.path.join(image_folder, image)))

# cv2.destroyAllWindows()
# video.release()