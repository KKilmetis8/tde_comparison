#!/bin/bash
radius=0.47
starmass=0.5
bhmass=10000 # don't use scientific notation here.
suffix="beta1S60n1.5Compton"

simulation="R"$radius"M"$starmass"BH"$bhmass""$suffix"" 

# Specify the desired snapsots, check which are available with rclone lsl
first_snap=80
last_snap=348
for  (( d=$first_snap; d<=$last_snap; d++ )) 
do
    # For downloading specific snapshots. Slow.
    # rclone copy gdrive:/Snellius/$simulation/snap_full_"$d".h5 . -P 
    mkdir $simulation/snap_"$d"
    mv snap_full_"$d".h5 "$simulation"/snap_"$d"/snap_"$d".h5                                                       
done         