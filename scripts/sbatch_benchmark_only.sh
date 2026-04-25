#!/bin/bash
# Re-run only the benchmark step for an already-trained set. Useful
# when the regen sbatch crashed at benchmark or we want to re-evaluate
# with different gate settings. Set SET, VERSION env vars.
#SBATCH --job-name=capiti-bench
#SBATCH --partition=priority_gpu
#SBATCH --account=prio_skr2
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/bench-%j.out
#SBATCH --error=logs/bench-%j.err

set -eo pipefail
mkdir -p logs

SET="${SET:-ab9}"
VERSION="${VERSION:-v1}"
ROOT_DATA="data/${SET}"
[ "$SET" = "ab9" ] && ROOT_DATA="data"

DATASET="${ROOT_DATA}/dataset/dataset.tsv"
PRIMARY="${ROOT_DATA}/targets/primary_sequences"
ACTIVE="${ROOT_DATA}/targets/active_sites"
RMAP="${ROOT_DATA}/targets/residue_maps"
OUT="${ROOT_DATA}/runs/${VERSION}"

module load miniconda
set +u
source /apps/software/system/software/miniconda/24.11.3/etc/profile.d/conda.sh
conda activate $HOME/project_pi_skr2/mcn26/capiti/.envs/baselines
set -u

SKIP="${SKIP:-}"
python -m src.eval.benchmark \
    --dataset "${DATASET}" \
    --targets "${PRIMARY}" \
    --capiti-onnx "${OUT}/capiti.onnx" \
    --capiti-meta "${OUT}/capiti.meta.json" \
    --capiti-name "capiti_${SET}_${VERSION}" \
    --out-dir "${OUT}/benchmark" \
    --active-sites "${ACTIVE}" \
    --residue-maps "${RMAP}" \
    --gate --gate-conf 0 ${SKIP:+--skip $SKIP}

DOCS="docs/benchmark/${SET}_${VERSION}"
mkdir -p "${DOCS}"
cp "${OUT}/benchmark/report.md" \
   "${OUT}/benchmark/metrics.json" \
   "${OUT}/benchmark/thresholds.json" \
   "${OUT}/benchmark"/*.png \
   "${DOCS}/"
cat "${OUT}/benchmark/report.md"
