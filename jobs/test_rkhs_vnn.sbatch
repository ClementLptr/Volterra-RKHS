#!/bin/bash 

#SBATCH --job-name=volterratrainning
#SBATCH --nodes=2
#SBATCH --partition=gpu_prod_long
#SBATCH --time=7:50:00

source /usr/users/volterrakernel/lepretre_cle/miniconda3/etc/profile.d/conda.sh activate env

export PYTHONPATH=/usr/users/volterrakernel/lepretre_cle/projet_volterra:$PYTHONPATH

# Run the Python script
python3 network/rgb_of/vnn_rgb_of_RKHS.py 
