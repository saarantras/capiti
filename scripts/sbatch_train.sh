#!/bin/bash
#SBATCH --job-name=capiti-train
#SBATCH --partition=priority_gpu
#SBATCH --account=prio_skr2
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err

set -eo pipefail
mkdir -p logs

module load miniconda
set +u
source /apps/software/system/software/miniconda/24.11.3/etc/profile.d/conda.sh
conda activate $HOME/project_pi_skr2/mcn26/capiti/.envs/esmfold
set -u

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m src.student.train --out-dir data/runs/v1 --epochs 15
