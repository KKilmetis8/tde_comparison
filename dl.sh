#!/bin/bash
# Place and run this in the directory where you want the files to be downloaded (for m=4 use the folder R1M1BH10000beta1)

# Download
for d in 116:197 # Change the range to the one you want
do
    rclone copy drive:/TEMPTDE4_new/snap_"$d".h5 . -P 
                                                                     
done                                                                 

# Make orderly
for d in 116:197 # Again, these are snapshot numbers
do
    mkdir "$d"
    mv snap_"$d".h5 "$d"/snap_"$d".h5
done

echo "Job's done!"

