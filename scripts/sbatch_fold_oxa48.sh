#!/bin/bash
#SBATCH --job-name=fold-oxa48
#SBATCH --partition=priority_gpu
#SBATCH --account=prio_skr2
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/fold-oxa48-%j.out
#SBATCH --error=logs/fold-oxa48-%j.err

set -eo pipefail
mkdir -p logs

module load miniconda
set +u
conda activate $HOME/project_pi_skr2/mcn26/capiti/.envs/esmfold
set -u

export HF_HOME=$HOME/project_pi_skr2/mcn26/capiti/.cache/hf
export HF_HUB_OFFLINE=1   # weights are cached; do not hit the network

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

IN=data/oracle_folds/oxa48_poster/inputs
OUT=data/oracle_folds/oxa48_poster/folds

python src/oracle/fold_esmfold.py \
    --fasta "$IN"/*.fasta \
    --out "$OUT"
