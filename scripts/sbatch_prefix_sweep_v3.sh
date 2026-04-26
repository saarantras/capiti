#!/bin/bash
# Run the prefix sweep against the existing v3 (no prefix-aug) ab9
# model so we have a "before" baseline to compare v4 against.
#SBATCH --job-name=capiti-sweep-v3
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=logs/sweep-v3-%j.out
#SBATCH --error=logs/sweep-v3-%j.err

set -eo pipefail
mkdir -p logs

module load miniconda
set +u
source /apps/software/system/software/miniconda/24.11.3/etc/profile.d/conda.sh
conda activate $HOME/project_pi_skr2/mcn26/capiti/.envs/baselines
set -u

python scripts/run_prefix_sweep.py \
    --dataset data/dataset/dataset.tsv \
    --targets data/targets/primary_sequences \
    --capiti-onnx data/runs/v3/capiti.onnx \
    --capiti-meta data/runs/v3/capiti.meta.json \
    --capiti-name capiti_ab9_v3 \
    --active-sites data/targets/active_sites \
    --residue-maps data/targets/residue_maps \
    --out-dir data/runs/v3/prefix_sweep

echo "v3 sweep -> data/runs/v3/prefix_sweep/"
