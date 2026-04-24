#!/bin/bash
# Train + benchmark three CapitiCNN variants in parallel:
#   1: maxpool  (--pool mean_max)
#   2: auxhead  (--aux-weight 0.3)
#   3: both     (--pool mean_max --aux-weight 0.3)
# Compare against the v2 baseline (plain mean pool, no aux) to see which
# lever moves the ala_scan number.
#
# Submit:  sbatch scripts/sbatch_variants.sh
#SBATCH --job-name=capiti-variants
#SBATCH --partition=priority_gpu
#SBATCH --account=prio_skr2
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --array=1-3
#SBATCH --output=logs/variant-%A_%a.out
#SBATCH --error=logs/variant-%A_%a.err

set -eo pipefail
mkdir -p logs

module load miniconda
set +u
source /apps/software/system/software/miniconda/24.11.3/etc/profile.d/conda.sh
conda activate $HOME/project_pi_skr2/mcn26/capiti/.envs/esmfold
set -u

case "$SLURM_ARRAY_TASK_ID" in
    1) NAME=maxpool;         POOL=mean_max; AUX=0   ;;
    2) NAME=auxhead;         POOL=mean;     AUX=0.3 ;;
    3) NAME=maxpool_auxhead; POOL=mean_max; AUX=0.3 ;;
    *) echo "bad array id"; exit 1 ;;
esac

OUT="data/runs/v2_${NAME}"
echo "=== $(date) ${NAME} on $(hostname) pool=${POOL} aux=${AUX} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 1. train
python -m src.student.train --out-dir "${OUT}" --epochs 15 \
    --pool "${POOL}" --aux-weight "${AUX}"

# 2. export ONNX (aux head stripped at export)
python -m src.student.export_onnx \
    --ckpt "${OUT}/best.pt" \
    --out-onnx "${OUT}/capiti.onnx" \
    --out-meta "${OUT}/capiti.meta.json" \
    --pool "${POOL}"

# 3. benchmark (swap to baselines env for blast/sklearn/onnxruntime)
set +u
conda activate $HOME/project_pi_skr2/mcn26/capiti/.envs/baselines
set -u
python -m src.eval.benchmark \
    --dataset data/dataset/dataset.tsv \
    --targets data/targets/primary_sequences \
    --capiti-onnx "${OUT}/capiti.onnx" \
    --capiti-meta "${OUT}/capiti.meta.json" \
    --capiti-name "capiti_v2_${NAME}" \
    --out-dir "${OUT}/benchmark"

# publish report + plots under docs/
DOCS="docs/benchmark/v2_${NAME}"
mkdir -p "${DOCS}"
cp "${OUT}/benchmark/report.md" \
   "${OUT}/benchmark/metrics.json" \
   "${OUT}/benchmark/thresholds.json" \
   "${OUT}/benchmark"/*.png \
   "${DOCS}/"

echo "=== $(date) ${NAME} done ==="
cat "${OUT}/benchmark/report.md"
