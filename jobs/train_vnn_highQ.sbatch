#!/bin/bash 

#SBATCH --job-name=volterratrainning
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=7:50:00

source /usr/users/volterrakernel/lepretre_cle/miniconda3/etc/profile.d/conda.sh activate env

# Run the Python script
python3 train_VNN_fusion_highQ.py
