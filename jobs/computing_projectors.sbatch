#!/bin/bash

#SBATCH --job-name=projector_computation
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=2-00:00:00  # 7 jours
#SBATCH --array=1-50       # Ajusté pour plus de combinaisons (exemple : 5 couches * 10 valeurs de P)
#SBATCH --output=logs/projectors/projector_layer_%a_P_%j_%A.out
#SBATCH --error=logs/projectors/projector_layer_%a_P_%j_%A.err


# Exit en cas d'erreur
set -e

# Charger l'environnement conda
source /usr/users/volterrakernel/lepretre_cle/miniconda3/etc/profile.d/conda.sh activate env

# Définir les paramètres
P_VALUES=( 24 48 96 192 384 512 640 768 896 1024 )  # Plage de 24 à 1024
TASK_ID=$SLURM_ARRAY_TASK_ID
LAYER=$(( (TASK_ID - 1) / 10 + 1 ))  # Couches 1-5 (10 P par couche)
P_IDX=$(( (TASK_ID - 1) % 10 ))      # Indice pour les valeurs de P (0-9)
P=${P_VALUES[$P_IDX]}

# Créer le répertoire de résultats s'il n'existe pas
mkdir -p results/projectors

# Journaliser les paramètres
echo "Démarrage de la tâche $TASK_ID : Couche $LAYER, P = $P"

# Exécuter le script Python avec vérification d'erreur
python compute_projectors.py \
    --layer "$LAYER" \
    --P "$P" \
    --save_dir results/projectors || { echo "Échec du script Python pour Couche $LAYER, P = $P"; exit 1; }

echo "Tâche $TASK_ID terminée : Couche $LAYER, P = $P"