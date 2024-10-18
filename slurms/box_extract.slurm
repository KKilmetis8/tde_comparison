#!/bin/bash
#SBATCH --job-name=box-6
#SBATCH --output=%x.out
#SBATCH --mail-user="kilmetis@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --partition="testing"
#AEK SBATCH --account="gpu_strw"
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --mem=32G

module load Python/3.10.4-GCCcore-11.3.0

# Place and run this in the directory where you want the files to be downloaded
# Specify the simulation to be downloaded
radius=0.47
starmass=0.5
bhmass=10000 # don't use scientific notation here.
suffix="beta1S60n1.5ComptonHiRes"

simulation="R"$radius"M"$starmass"BH"$bhmass""$suffix"" 

# Specify the desired snapsots, check which are available with rclone lsl
first_snap=200
last_snap=300

# Extract data from .h5 to .npy
cd tde_comparison
python3 -m src.Extractors.boxextract --name $simulation \
 --mass $starmass --radius $radius --blackhole $bhmass \
 --first $first_snap --last $last_snap -s False

echo "Job's done!"