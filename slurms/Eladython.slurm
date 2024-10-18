#!/bin/bash
#SBATCH --job-name=sp6
#SBATCH --output=%x.out


#SBATCH --mail-user="kilmetis@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --partition="cpu-long"
#aek SBATCH --account="gpu_strw"
#SBATCH --time=3-0:00:00
#SBATCH --ntasks=1
#SBATCH --mem=100G

# Load python
module load Python/3.11.3-GCCcore-12.3.0
source ~/data1/python_venvs/env3/bin/activate
module load MATLAB

export LD_LIBRARY_PATH=/easybuild/software/MATLAB/2023b-r4/bin/glnxa64:${LD_LIBRARY_PATH}

pip install matlabengine==23.2.3

# Load MATLAB
module load MATLAB
pip install healpy
#export LD_LIBRARY_PATH=/easybuild/software/MATLAB/2023b-r4/bin/ginxa64:{$LD_LIBRARY_PATH}

# Specify the simulation to be used
radius=0.47
starmass=0.5
bhmass=1e+06 # don't use scientific notation here.
suffix="beta1S60n1.5Compton"

simulation="R"$radius"M"$starmass"BH"$bhmass""$suffix"" 

# Specify the desired snapsots
first_snap=186 # 4f 80(140) 4h 101 4s 120 5 130 6 132 (180)
last_snap=365 # 4f 348 4h 216 4s 172 5 365 6 365

echo "[$SHELL] #### Starting Python Job"
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME and this job has the ID $SLURM_JOB_ID"
# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD

# Run the file
echo "[$SHELL] ## Run script"
cd tde_comparison
python3 -m src.Luminosity.Eladython --name $simulation \
 --mass $starmass --radius $radius --blackhole $bhmass \
 --first $first_snap --last $last_snap
echo "[$SHELL] ## Script finished"
