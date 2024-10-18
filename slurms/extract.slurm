#!/bin/bash
#SBATCH --job-name=e5
#SBATCH --output=%x.out
#SBATCH --mail-user="kilmetis@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --partition="gpu_strw"
##SBATCH --account="gpu_strw"
#SBATCH --time=1-0:00:00
#SBATCH --ntasks=1
#SBATCH --mem=30G

module load Python/3.10.4-GCCcore-11.3.0

# Place and run this in the directory where you want the files to be downloaded
# Specify the simulation to be downloaded
radius=0.47
starmass=0.5
bhmass=100000 # don't use scientific notation here.
suffix="beta1S60n1.5Compton"

simulation="R"$radius"M"$starmass"BH"$bhmass""$suffix"" 

# Specify the desired snapsots, check which are available with rclone lsl
first_snap=80
last_snap=403

# Extract data from .h5 to .npy
python tde_comparison/src/Extractors/gen_extractor.py --name $simulation \
 --mass $starmass --radius $radius --blackhole $bhmass \
 --first $first_snap --last $last_snap

echo "Job's done!"