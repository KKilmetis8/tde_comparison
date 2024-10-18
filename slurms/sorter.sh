#!/bin/bash

module load Python/3.10.4-GCCcore-11.3.0
module load rclone

# Place and run this in the directory where you want the files to be downloaded
# Specify the simulation to be downloaded
radius=0.47
starmass=0.5
bhmass=100000 # don't use scientific notation here.
suffix="beta1S60n1.5Compton"

simulation="R"$radius"M"$starmass"BH"$bhmass""$suffix"" 

# Specify the desired snapsots, check which are available with rclone lsl
first_snap=130
last_snap=270

# Download & place into seperete dirs.
for  (( d=$first_snap; d<=$last_snap; d++ )) 
do
    # For downloading specific snapshots. Slow.
    # rclone copy gdrive:/Snellius/$simulation/snap_full_"$d".h5 . -P 
    # mkdir $simulation/snap_"$d"
    mv "$simulation"/snap_"$d".h5 "$simulation"/snap_"$d"/snap_"$d".h5                                                       
done   