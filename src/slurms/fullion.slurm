#!/bin/bash
#SBATCH --job-name=ion-4s
#SBATCH --output=%x.out
#SBATCH --mail-user="kilmetis@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --partition="gpu_strw"
#SBATCH --account="gpu_strw"
#SBATCH --time=1-0:00:00
#SBATCH --ntasks=1
#SBATCH --mem=24G

# Load python
module load Python/3.10.4-GCCcore-11.3.0
pip install tqdm
# Specify the simulation to be downloaded
radius=0.47
starmass=0.5
bhmass=10000 # don't use scientific notation here.
suffix="beta1S60n1.5ComptonRes20"

simulation="R"$radius"M"$starmass"BH"$bhmass""$suffix"" 

# Specify the desired snapsots, check which are available with rclone lsl
first_snap=80 # 4f 80(140) 4h 80(100) 4s 120 5 250
last_snap=348 # 4f 348 4h 216 4s 172 5 365

echo "[$SHELL] #### Starting Python Job"
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME and this job has the ID $SLURM_JOB_ID"
# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD

# Run the file
echo "[$SHELL] ## Run script"
cd tde_comparison
python3 -m src.IonizationShells.fullion_inter  --name $simulation \
 --mass $starmass --radius $radius --blackhole $bhmass \
 --first $first_snap --last $last_snap
echo "[$SHELL] ## Script finished"

