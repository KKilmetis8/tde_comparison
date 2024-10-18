#!/bin/bash
#SBATCH --job-name=as6
#SBATCH --output=%x.out
#SBATCH --mail-user="kilmetis@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --partition="gpu_strw"
#SBATCH --account="gpu_strw"
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=20G

# Specify the desired snapsots
first_snap=180 # 4f 80(140) 4h 101 4s 120 5 130 6 132 (180)
last_snap=403 # 4f 348 4h 216 4s 172 5 365 6 365

# Load python
module load Python/3.11.3-GCCcore-12.3.0
source ~/data1/python_venvs/env3/bin/activate

# Specify the simulation to be used
radius=0.47
starmass=0.5
bhmass=1e+06 # don't use scientific notation here.
suffix="beta1S60n1.5Compton"

simulation="R"$radius"M"$starmass"BH"$bhmass""$suffix"" 

echo "[$SHELL] #### Starting Python Job"
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME and this job has the ID $SLURM_JOB_ID"
# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD

# Run the file
echo "[$SHELL] ## Run script"
cd tde_comparison

for  (( d=$first_snap; d<=$last_snap; d++ )) 
do
    python3 -m src.Eccentricity.actionspace --name $simulation \
    --mass $starmass --radius $radius --blackhole $bhmass \
    --first $first_snap --last $last_snap --single True \
    --only $d
done
echo "[$SHELL] ## Script finished"
