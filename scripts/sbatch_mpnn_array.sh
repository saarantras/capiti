#!/bin/bash
#SBATCH --job-name=mpnn-gen
#SBATCH --partition=priority_gpu
#SBATCH --account=prio_skr2
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=1-9
#SBATCH --output=logs/mpnn-gen-%A_%a.out
#SBATCH --error=logs/mpnn-gen-%A_%a.err

set -eo pipefail
mkdir -p logs

module load miniconda
set +u
source /apps/software/system/software/miniconda/24.11.3/etc/profile.d/conda.sh
conda activate $HOME/project_pi_skr2/mcn26/capiti/.envs/esmfold
set -u

TID="T${SLURM_ARRAY_TASK_ID}"
echo "=== $TID on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python src/data/generate_mpnn_variants.py --target "$TID"
