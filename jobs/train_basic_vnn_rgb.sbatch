#!/bin/bash 

#SBATCH --job-name=volterratrainning
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=7:50:00
#SBATCH --output=logs/model_training/basic_rgb_7Q_%A.out
#SBATCH --error=logs/model_training/basic_rgb_7Q_%A.err

source /usr/users/volterrakernel/lepretre_cle/miniconda3/etc/profile.d/conda.sh activate env

# Run the Python script
python3 train_basic_VNN_RGB.py
