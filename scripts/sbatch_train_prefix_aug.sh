#!/bin/bash
# Train + export + prefix-sweep for the streaming-inference student.
#
# Mirrors sbatch_regen_downstream.sh's path conventions but skips
# negatives + dataset assembly (re-uses the already-built dataset for
# the chosen SET) and adds a final prefix sweep step.
#
# Env vars:
#   SET      ab9 | C | E   default ab9
#   VERSION  run-dir name  default v4_prefixaug
#   EPOCHS   training      default 25
#   PREFIX_KEEP_FULL        default 0.5
#   PREFIX_MIN_FRAC         default 0.40
#
# Submit:
#   sbatch --export=ALL,SET=ab9,VERSION=v4_prefixaug \
#       scripts/sbatch_train_prefix_aug.sh
#SBATCH --job-name=capiti-prefixaug
#SBATCH --partition=priority_gpu
#SBATCH --account=prio_skr2
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:30:00
#SBATCH --output=logs/prefixaug-%j.out
#SBATCH --error=logs/prefixaug-%j.err

set -eo pipefail
mkdir -p logs

SET="${SET:-ab9}"
VERSION="${VERSION:-v4_prefixaug}"
EPOCHS="${EPOCHS:-25}"
PREFIX_KEEP_FULL="${PREFIX_KEEP_FULL:-0.5}"
PREFIX_MIN_FRAC="${PREFIX_MIN_FRAC:-0.40}"

if [ "$SET" = "ab9" ]; then
    PRIMARY="data/targets/primary_sequences"
    ACTIVE="data/targets/active_sites"
    RMAP="data/targets/residue_maps"
    DATASET="data/dataset"
    RUNS="data/runs"
else
    ROOT_DATA="data/${SET}"
    PRIMARY="${ROOT_DATA}/targets/primary_sequences"
    ACTIVE="${ROOT_DATA}/targets/active_sites"
    RMAP="${ROOT_DATA}/targets/residue_maps"
    DATASET="${ROOT_DATA}/dataset"
    RUNS="${ROOT_DATA}/runs"
fi
OUT="${RUNS}/${VERSION}"

module load miniconda
set +u
source /apps/software/system/software/miniconda/24.11.3/etc/profile.d/conda.sh
conda activate $HOME/project_pi_skr2/mcn26/capiti/.envs/esmfold
set -u

echo "=== $(date): prefixaug set=${SET} version=${VERSION} on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 1. train with prefix augmentation
echo "--- train ---"
python -m src.student.train \
    --dataset "${DATASET}/dataset.tsv" \
    --out-dir "${OUT}" \
    --epochs "${EPOCHS}" \
    --pool mean_max \
    --residue-maps "${RMAP}" \
    --active-sites "${ACTIVE}" \
    --prefix-aug \
    --prefix-keep-full "${PREFIX_KEEP_FULL}" \
    --prefix-min-frac "${PREFIX_MIN_FRAC}"

# 2. export ONNX
echo "--- export_onnx ---"
python -m src.student.export_onnx \
    --ckpt "${OUT}/best.pt" \
    --out-onnx "${OUT}/capiti.onnx" \
    --out-meta "${OUT}/capiti.meta.json" \
    --pool mean_max

# 3. prefix sweep (uses the baselines env for blast/sklearn)
echo "--- prefix sweep ---"
set +u
conda activate $HOME/project_pi_skr2/mcn26/capiti/.envs/baselines
set -u
python scripts/run_prefix_sweep.py \
    --dataset "${DATASET}/dataset.tsv" \
    --targets "${PRIMARY}" \
    --capiti-onnx "${OUT}/capiti.onnx" \
    --capiti-meta "${OUT}/capiti.meta.json" \
    --capiti-name "capiti_${SET}_${VERSION}" \
    --active-sites "${ACTIVE}" \
    --residue-maps "${RMAP}" \
    --out-dir "${OUT}/prefix_sweep"

DOCS="docs/benchmark/${SET}_${VERSION}_prefix_sweep"
mkdir -p "${DOCS}"
cp "${OUT}/prefix_sweep"/*.png \
   "${OUT}/prefix_sweep/per_target_min_prefix.tsv" \
   "${OUT}/prefix_sweep/prefix_metrics.tsv" \
   "${DOCS}/"

echo "=== $(date): ${SET}/${VERSION} done ==="
echo "outputs: ${OUT}/prefix_sweep/  +  ${DOCS}/"
