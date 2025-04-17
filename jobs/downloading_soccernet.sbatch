#!/bin/bash 

#SBATCH --job-name=volterra
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=7:00:00

source /usr/users/volterrakernel/lepretre_cle/miniconda3/etc/profile.d/conda.sh activate env

# Run the Python script
python3 config/dataloaders/soccernet.py
