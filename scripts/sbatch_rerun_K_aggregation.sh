#!/bin/bash
# CPU-only re-aggregation: regenerate K-binned outputs for existing
# sweep dirs, since the per-frac scores.tsv files are already cached.
#SBATCH --job-name=capiti-K-agg
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/K-agg-%j.out
#SBATCH --error=logs/K-agg-%j.err

set -eo pipefail
mkdir -p logs

module load miniconda
set +u
source /apps/software/system/software/miniconda/24.11.3/etc/profile.d/conda.sh
conda activate $HOME/project_pi_skr2/mcn26/capiti/.envs/baselines
set -u

for run in v4_prefixaug v4_aug_c0_baseline v4_aug_c1_morepartial \
           v4_aug_c2_shorter v4_aug_c3_aggressive v4_aug_c4; do
    OUT="data/runs/${run}/prefix_sweep"
    [ -d "$OUT" ] || { echo "skip $run (no sweep dir)"; continue; }
    [ -f "${OUT}/frac_1.00/scores.tsv" ] || { echo "skip $run (no cached frac_1.00)"; continue; }
    echo "--- ${run} ---"
    python scripts/run_prefix_sweep.py \
        --dataset data/dataset/dataset.tsv \
        --targets data/targets/primary_sequences \
        --capiti-onnx "data/runs/${run}/capiti.onnx" \
        --capiti-meta "data/runs/${run}/capiti.meta.json" \
        --capiti-name "capiti_ab9_${run}" \
        --active-sites data/targets/active_sites \
        --residue-maps data/targets/residue_maps \
        --out-dir "${OUT}" \
        --k-bin-width 100
done

echo "=== per_target_min_K (v4_aug_c1_morepartial) ==="
cat data/runs/v4_aug_c1_morepartial/prefix_sweep/per_target_min_K.tsv | grep -v blast
echo
echo "=== threshold schedule (capiti, c1) ==="
awk -F'\t' '$1 ~ /capiti/ || NR==1 {print}' \
    data/runs/v4_aug_c1_morepartial/prefix_sweep/threshold_schedule_by_K.tsv
