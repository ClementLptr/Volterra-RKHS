#!/bin/bash 

#SBATCH --job-name=volterratrainning
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=20:00:00
#SBATCH --output=logs/model_training/train_CNN_toy_dataset%A.out
#SBATCH --error=logs/model_training/train_CNN_toy_dataset%A.err

source /usr/users/volterrakernel/lepretre_cle/miniconda3/etc/profile.d/conda.sh activate env

# Run the Python script
python3 train_CNN_toy_dataset.py --num_projectors 24