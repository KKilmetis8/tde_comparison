#!/bin/bash
#SBATCH --job-name=sp-p-5
#SBATCH --output=%x.out
#SBATCH --mail-user="kilmetis@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --partition="testing"
#AEK SBATCH --account="gpu_strw"
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --mem=10G

# Load python
module load Python/3.11.3-GCCcore-12.3.0
source ~/data1/python_venvs/env3/bin/activate
# pip list

# Load MATLAB
module load MATLAB
export LD_LIBRARY_PATH=/easybuild/software/MATLAB/2023b-r4/bin/glnxa64:${LD_LIBRARY_PATH}
pip install matlabengine==23.2.3

# Packages
# pip install numpy
# pip install healpy
# pip install tqdm
# pip install scikit-learn

# Specify the simulation to be used
radius=0.47
starmass=0.5
bhmass=100000 # don't use scientific notation here.
suffix="beta1S60n1.5Compton"
simulation="R"$radius"M"$starmass"BH"$bhmass""$suffix"" 

# Specify the desired snapsots
first_snap=193 # 4f 80(140) 4h 101 4s 120 5 130 6 132 (180)
last_snap=210 # 4f 348 4h 216 4s 172 5 365 6 365

echo "[$SHELL] #### Starting Python Job"
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME and this job has the ID $SLURM_JOB_ID"
# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD

# Run the file
echo "[$SHELL] ## Run script"
cd tde_comparison
python3 -m src.Luminosity.spectra_Elad --name $simulation \
 --mass $starmass --radius $radius --blackhole $bhmass \
 --first $first_snap --last $last_snap
echo "[$SHELL] ## Script finished"
