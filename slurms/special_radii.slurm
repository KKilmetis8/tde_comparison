#!/bin/bash
#SBATCH --job-name=spec_radii
#SBATCH --output=%x.out
#SBATCH --mail-user="martire@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --partition="gpu_strw"
#SBATCH --account="gpu_strw"
#SBATCH --time=6-0:00:00
#SBATCH --ntasks=1
#SBATCH --mem=16G

# Load python
module load ALICE/AMD
module load Python/3.10.4-GCCcore-11.3.0

# Start the env
source python_venvs/amd_base/bin/activate	


echo "[$SHELL] #### Starting Python Job"
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME and this job has the ID $SLURM_JOB_ID"
# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD

# Run the file
echo "[$SHELL] ## Run script"
cd TDE/tde_comparison
python3 -m src.Luminosity.special_radii_tree
echo "[$SHELL] ## Script finished"
