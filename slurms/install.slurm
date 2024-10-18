#!/bin/bash
#SBATCH --job-name=install
#SBATCH --output=%x.out
#SBATCH --mail-user="kilmetis@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --partition="gpu_strw"
#SBATCH --account="gpu_strw"
#SBATCH --time=1-0:00:00
#SBATCH --ntasks=1
#SBATCH --mem=32G

# Load python
module load ALICE/default
module load Python/3.11.3-GCCcore-12.3.0
module load MATLAB

# Start the env
#source python_venvs/amd_base2/bin/activate
export LD_LIBRARY_PATH=/easybuild/software/MATLAB/2023b-r4/bin/ginxa64
pip install matlabengine==23.2.3
pip install scipy
pip install numba
pip install matplotlib
pip install colorcet
pip install pandas
pip install numpy
