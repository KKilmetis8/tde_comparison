#!/bin/bash
#SBATCH --job-name=diss6pe
#SBATCH --output=%x.out
#SBATCH --mail-user="kilmetis@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --partition="gpu_strw"
#SBATCH --account="gpu_strw"
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --mem=32G

# Load python
module load Python/3.10.4-GCCcore-11.3.0
# Specify the simulation to be downloaded
radius=0.47
starmass=0.5
bhmass=1e+06 # don't use scientific notation here.
suffix="beta1S60n1.5Compton"
simulation="R"$radius"M"$starmass"BH"$bhmass""$suffix"" 

what=Diss
# Specify the desired snapsots
first_snap=351 # 4f 80(140) 4h 80(100) 4s 120 5 250 6 132 (180)
last_snap=351 # 4f 348 4h 216 4s 172 5 365 6 365

echo "[$SHELL] #### Starting Python Job"
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME and this job has the ID $SLURM_JOB_ID"
# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD

# Run the file
echo "[$SHELL] ## Run script"
cd tde_comparison
python3 -m src.Projectors.projector_tree  --name $simulation \
 --mass $starmass --radius $radius --blackhole $bhmass \
 --first $first_snap --last $last_snap --what $what
echo "[$SHELL] ## Script finished"

