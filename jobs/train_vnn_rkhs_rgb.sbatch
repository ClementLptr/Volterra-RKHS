#!/bin/bash 

#SBATCH --job-name=volterratrainning
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=20:00:00
#SBATCH --output=logs/model_training/24_p_rkhs_training_%A.out
#SBATCH --error=logs/model_training/24_p_rkhs_training_%A.err

source /usr/users/volterrakernel/lepretre_cle/miniconda3/etc/profile.d/conda.sh activate env

# Run the Python script
python3 train_VNN_RKHS_RGB.py --num_projectors 24