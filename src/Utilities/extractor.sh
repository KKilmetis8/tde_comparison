#!/bin/bash
radius=0.47
starmass=0.5
bhmass=10000 # don't use scientific notation here.
suffix="beta1S60n1.5Compton"

simulation="R"$radius"M"$starmass"BH"$bhmass""$suffix"" 
# Specify the desired snapsots, check which are available with rclone lsl
first_snap=80
last_snap=348
# Extract data from .h5 to .npy
python tde_comparison/src/Extractors/super_extractor.py --name $simulation \
 --mass $starmass --radius $radius --blackhole $bhmass \
 --first $first_snap --last $last_snap

echo "Job's done!"