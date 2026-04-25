#!/bin/bash
# Per-set downstream pipeline after MPNN generation:
#   1. generate_negatives.py for each target in configs/targets-<SET>.tsv
#   2. assemble_dataset.py -> per-set dataset.tsv
#   3. src.student.train -> data/<set-root>/runs/<VERSION>/best.pt
#   4. export_onnx -> data/<set-root>/runs/<VERSION>/capiti.{onnx,meta.json}
#   5. bundle into capiti/_model/<SET>/ (overwrite)
#   6. src.eval.benchmark with --gate -> benchmark artifacts + docs/
#
# Env vars:
#   SET      ab9 | C | E        default ab9
#   VERSION  run-dir name       default v3  (per-set: use VERSION=v1 for first run of a new set)
#
# Submit:
#   mpnn_id=$(sbatch --parsable --export=ALL,SET=E --array=1-59%30 scripts/sbatch_mpnn_array.sh)
#   sbatch --dependency=afterok:$mpnn_id --export=ALL,SET=E,VERSION=v1 scripts/sbatch_regen_downstream.sh
#SBATCH --job-name=capiti-regen
#SBATCH --partition=priority_gpu
#SBATCH --account=prio_skr2
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=logs/regen-%j.out
#SBATCH --error=logs/regen-%j.err

set -eo pipefail
mkdir -p logs

SET="${SET:-ab9}"
VERSION="${VERSION:-v3}"

if [ "$SET" = "ab9" ]; then
    TARGETS_TSV="configs/targets.tsv"
    ROOT_DATA="data"  # implicit: data/targets, data/variants, data/dataset
    PRIMARY="data/targets/primary_sequences"
    ACTIVE="data/targets/active_sites"
    RMAP="data/targets/residue_maps"
    VARIANTS="data/variants"
    DATASET="data/dataset"
    RUNS="data/runs"
else
    TARGETS_TSV="configs/targets-${SET}.tsv"
    ROOT_DATA="data/${SET}"
    PRIMARY="${ROOT_DATA}/targets/primary_sequences"
    ACTIVE="${ROOT_DATA}/targets/active_sites"
    RMAP="${ROOT_DATA}/targets/residue_maps"
    VARIANTS="${ROOT_DATA}/variants"
    DATASET="${ROOT_DATA}/dataset"
    RUNS="${ROOT_DATA}/runs"
fi
OUT="${RUNS}/${VERSION}"

module load miniconda
set +u
source /apps/software/system/software/miniconda/24.11.3/etc/profile.d/conda.sh
conda activate $HOME/project_pi_skr2/mcn26/capiti/.envs/esmfold
set -u

echo "=== $(date): regen set=${SET} version=${VERSION} on $(hostname) ==="

# 1. negatives (skipping targets already done so we can iterate cheaply)
mkdir -p "${VARIANTS}"
while read -r tid _; do
    [[ -z "$tid" || "$tid" == \#* ]] && continue
    if [ -f "${VARIANTS}/ala_scan/${tid}.fasta" ] && \
       [ -f "${VARIANTS}/combined_ko/${tid}.fasta" ] && \
       [ -f "${VARIANTS}/scramble/${tid}.fasta" ] && \
       [ -f "${VARIANTS}/perturb30/${tid}.fasta" ]; then
        continue
    fi
    echo "--- generate_negatives ${tid} ---"
    python -m src.data.generate_negatives \
        --target "${tid}" \
        --primary-seqs "${PRIMARY}" \
        --active-sites "${ACTIVE}" \
        --residue-maps "${RMAP}" \
        --mpnn-fasta "${VARIANTS}/mpnn_positives" \
        --out-root "${VARIANTS}"
done < "${TARGETS_TSV}"

# 2. assemble dataset
echo "--- assemble_dataset ---"
python -m src.data.assemble_dataset \
    --out-dir "${DATASET}" \
    --primary-seqs "${PRIMARY}" \
    --variants "${VARIANTS}" \
    --targets-tsv "${TARGETS_TSV}"

# 3. train
echo "--- train ---"
python -m src.student.train \
    --dataset "${DATASET}/dataset.tsv" \
    --out-dir "${OUT}" \
    --epochs 15 \
    --pool mean_max \
    --residue-maps "${RMAP}" \
    --active-sites "${ACTIVE}"

# 4. export ONNX
echo "--- export_onnx ---"
python -m src.student.export_onnx \
    --ckpt "${OUT}/best.pt" \
    --out-onnx "${OUT}/capiti.onnx" \
    --out-meta "${OUT}/capiti.meta.json" \
    --pool mean_max

# 5. bundle
BUNDLE_DIR="capiti/_model/${SET}"
mkdir -p "${BUNDLE_DIR}"
cp "${OUT}/capiti.onnx" "${BUNDLE_DIR}/capiti.onnx"
cp "${OUT}/capiti.meta.json" "${BUNDLE_DIR}/capiti.meta.json"
echo "bundled -> ${BUNDLE_DIR}"

# 6. benchmark (gated)
echo "--- benchmark ---"
set +u
conda activate $HOME/project_pi_skr2/mcn26/capiti/.envs/baselines
set -u
python -m src.eval.benchmark \
    --dataset "${DATASET}/dataset.tsv" \
    --targets "${PRIMARY}" \
    --capiti-onnx "${OUT}/capiti.onnx" \
    --capiti-meta "${OUT}/capiti.meta.json" \
    --capiti-name "capiti_${SET}_${VERSION}" \
    --out-dir "${OUT}/benchmark" \
    --active-sites "${ACTIVE}" \
    --residue-maps "${RMAP}" \
    --gate --gate-conf 0

DOCS="docs/benchmark/${SET}_${VERSION}"
mkdir -p "${DOCS}"
cp "${OUT}/benchmark/report.md" \
   "${OUT}/benchmark/metrics.json" \
   "${OUT}/benchmark/thresholds.json" \
   "${OUT}/benchmark"/*.png \
   "${DOCS}/"

echo "=== $(date): ${SET}/${VERSION} done ==="
cat "${OUT}/benchmark/report.md"
