#!/bin/bash
# 4-config prefix-aug hyperparameter sweep on ab9. Trains + sweeps each
# config sequentially on one GPU, then writes a combined comparison
# table of per-target min-prefix-for-TPR-target across configs.
#
# Configs:
#   c0_baseline       keep_full=0.5  min_frac=0.40  bias=uniform   (== current v4)
#   c1_morepartial    keep_full=0.2  min_frac=0.40  bias=uniform
#   c2_shorter        keep_full=0.5  min_frac=0.20  bias=uniform
#   c3_aggressive     keep_full=0.2  min_frac=0.20  bias=short
#
# Submit:
#   sbatch --gpus=h200:1 scripts/sbatch_prefix_aug_sweep.sh
#SBATCH --job-name=capiti-augsweep
#SBATCH --partition=priority_gpu
#SBATCH --account=prio_skr2
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=logs/augsweep-%j.out
#SBATCH --error=logs/augsweep-%j.err

set -eo pipefail
mkdir -p logs

DATASET="data/dataset/dataset.tsv"
PRIMARY="data/targets/primary_sequences"
ACTIVE="data/targets/active_sites"
RMAP="data/targets/residue_maps"
RUNS="data/runs"
EPOCHS=25

CONFIGS=(
    "c0_baseline:0.5:0.40:uniform"
    "c1_morepartial:0.2:0.40:uniform"
    "c2_shorter:0.5:0.20:uniform"
    "c3_aggressive:0.2:0.20:short"
)

module load miniconda
set +u
source /apps/software/system/software/miniconda/24.11.3/etc/profile.d/conda.sh
ENV_TRAIN="$HOME/project_pi_skr2/mcn26/capiti/.envs/esmfold"
ENV_BENCH="$HOME/project_pi_skr2/mcn26/capiti/.envs/baselines"
set -u

echo "=== $(date): augsweep on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for entry in "${CONFIGS[@]}"; do
    IFS=":" read -r name keep_full min_frac bias <<< "$entry"
    OUT="${RUNS}/v4_aug_${name}"
    echo
    echo "======================================================="
    echo "  config=${name}  keep_full=${keep_full}  min_frac=${min_frac}  bias=${bias}"
    echo "======================================================="

    set +u; conda activate "${ENV_TRAIN}"; set -u
    python -m src.student.train \
        --dataset "${DATASET}" \
        --out-dir "${OUT}" \
        --epochs "${EPOCHS}" \
        --pool mean_max \
        --residue-maps "${RMAP}" \
        --active-sites "${ACTIVE}" \
        --prefix-aug \
        --prefix-keep-full "${keep_full}" \
        --prefix-min-frac "${min_frac}" \
        --prefix-bias "${bias}"

    python -m src.student.export_onnx \
        --ckpt "${OUT}/best.pt" \
        --out-onnx "${OUT}/capiti.onnx" \
        --out-meta "${OUT}/capiti.meta.json" \
        --pool mean_max

    set +u; conda activate "${ENV_BENCH}"; set -u
    python scripts/run_prefix_sweep.py \
        --dataset "${DATASET}" \
        --targets "${PRIMARY}" \
        --capiti-onnx "${OUT}/capiti.onnx" \
        --capiti-meta "${OUT}/capiti.meta.json" \
        --capiti-name "capiti_ab9_${name}" \
        --active-sites "${ACTIVE}" \
        --residue-maps "${RMAP}" \
        --out-dir "${OUT}/prefix_sweep"
done

# Combined comparison table
echo
echo "=== combined comparison table ==="
python scripts/compare_aug_configs.py \
    --runs-root "${RUNS}" \
    --configs "${CONFIGS[@]}" \
    --out "${RUNS}/v4_augsweep_comparison.tsv"

echo "=== $(date): augsweep done ==="
cat "${RUNS}/v4_augsweep_comparison.tsv"
