#!/bin/bash
# Place and run this in the directory where you want the files to be downloaded

# Download
for d in {232..264} # Change the range to the one you want
do
    rclone copy gdrive:/Phys/sims/R1M1BH10000beta1/snap_"$d".h5 . -P 
                                                                     
done                                                                 

# Make orderly
for d in {232..264} # Again, these are snapshot numbers
do
    mkdir "$d"
    mv snap_"$d".h5 "$d"/snap_"$d".h5
done

echo "Job's done!"

