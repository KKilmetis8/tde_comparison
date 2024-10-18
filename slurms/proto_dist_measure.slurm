#!/bin/bash
#SBATCH --job-name=d<m>-<num>
#SBATCH --output=%x.out
#SBATCH --mail-user="kilmetis@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --partition="cpu-long"
#aek SBATCH --account="gpu_strw"
#SBATCH --time=4-0:00:00
#SBATCH --ntasks=1
#SBATCH --mem=15G

module load Python/3.10.4-GCCcore-11.3.0

# Place and run this in the directory where you want the files to be downloaded
# Specify the simulation to be downloaded
radius=0.47
starmass=0.5
bhmass=<bh> # don't use scientific notation here.
suffix=<suffix>

simulation="R"$radius"M"$starmass"BH"$bhmass""$suffix"" 

# Specify the desired snapsots, check which are available with rclone lsl
first_snap=<first> 
last_snap=<last>

# Extract data from .h5 to .npy
cd tde_comparison
for  (( d=$first_snap; d<=$last_snap; d++ )) 
do
    python3 -m src.ToyModel.distancemeasurer --name $simulation \
    --mass $starmass --radius $radius --blackhole $bhmass \
    --first $first_snap --last $last_snap -s False \
    --only $d
done

echo "Job's done!"