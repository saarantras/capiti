#!/bin/bash
# MPNN generation for one target set. SET env var selects which set:
#   SET=ab9  -> configs/targets.tsv,   data/targets/, data/variants/
#   SET=C    -> configs/targets-C.tsv, data/C/targets/, data/C/variants/
#   SET=E    -> configs/targets-E.tsv, data/E/targets/, data/E/variants/
#
# Array id is the 1-indexed line number in the targets TSV (skipping
# comment / blank lines). Submit with explicit array range sized to the
# config:
#   sbatch --export=ALL,SET=E --array=1-59%30 scripts/sbatch_mpnn_array.sh
#SBATCH --job-name=mpnn-gen
#SBATCH --partition=priority_gpu
#SBATCH --account=prio_skr2
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=00:45:00
#SBATCH --output=logs/mpnn-gen-%A_%a.out
#SBATCH --error=logs/mpnn-gen-%A_%a.err

set -eo pipefail
mkdir -p logs

SET="${SET:-ab9}"
if [ "$SET" = "ab9" ]; then
    TARGETS_TSV="configs/targets.tsv"
    STRUCT_DIR="data/targets/structures"
    ACTIVE_DIR="data/targets/active_sites"
    RMAP_DIR="data/targets/residue_maps"
    OUT_DIR="data/variants/mpnn_positives"
    WORK_DIR="data/variants/_mpnn_work"
else
    TARGETS_TSV="configs/targets-${SET}.tsv"
    ROOT="data/${SET}"
    STRUCT_DIR="${ROOT}/targets/structures"
    ACTIVE_DIR="${ROOT}/targets/active_sites"
    RMAP_DIR="${ROOT}/targets/residue_maps"
    OUT_DIR="${ROOT}/variants/mpnn_positives"
    WORK_DIR="${ROOT}/variants/_mpnn_work"
fi

# Resolve array id -> target id from the TSV (skip comments/blanks)
TID=$(awk '!/^#/ && NF' "$TARGETS_TSV" | awk -v n="$SLURM_ARRAY_TASK_ID" 'NR==n {print $1}')
if [ -z "$TID" ]; then
    echo "no target for array index $SLURM_ARRAY_TASK_ID in $TARGETS_TSV"
    exit 1
fi

module load miniconda
set +u
source /apps/software/system/software/miniconda/24.11.3/etc/profile.d/conda.sh
conda activate $HOME/project_pi_skr2/mcn26/capiti/.envs/esmfold
set -u

echo "=== set=$SET tid=$TID on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -m src.data.generate_mpnn_variants \
    --target "$TID" \
    --structures "$STRUCT_DIR" \
    --active-sites "$ACTIVE_DIR" \
    --residue-maps "$RMAP_DIR" \
    --out-dir "$OUT_DIR"
