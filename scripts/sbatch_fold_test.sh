#!/bin/bash
#SBATCH --job-name=fold-test
#SBATCH --partition=priority_gpu
#SBATCH --account=prio_skr2
#SBATCH --gres=gpu:rtx_5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/fold-test-%j.out
#SBATCH --error=logs/fold-test-%j.err

set -eo pipefail
mkdir -p logs

module load miniconda
set +u
conda activate $HOME/project_pi_skr2/mcn26/capiti/.envs/esmfold
set -u

export HF_HOME=$HOME/project_pi_skr2/mcn26/capiti/.cache/hf
mkdir -p "$HF_HOME"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python src/oracle/fold_esmfold.py \
    --fasta tracked_data/targets/primary_sequences/T2.fasta \
    --out data/oracle_folds/test
