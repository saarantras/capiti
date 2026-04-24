#!/bin/bash
# After MPNN regeneration finishes, this job:
#   1. generate_negatives.py for each Ti (WT + MPNN ala_scan, combined_ko, scramble, perturb30)
#   2. assemble_dataset.py (fresh dataset.tsv with stratified splits)
#   3. train (src.student.train) -> data/runs/v2/best.pt
#   4. export_onnx -> data/runs/v2/capiti.{onnx,meta.json}
#   5. bundle into capiti/_model/ab9/ (overwriting the v1 bundle)
#   6. run the benchmark pipeline -> data/runs/v2/benchmark/ + docs/benchmark/v2/
#
# Submit after the MPNN array:
#     mpnn_id=$(sbatch --parsable scripts/sbatch_mpnn_array.sh)
#     sbatch --dependency=afterok:$mpnn_id scripts/sbatch_regen_downstream.sh
#SBATCH --job-name=capiti-regen
#SBATCH --partition=priority_gpu
#SBATCH --account=prio_skr2
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/regen-%j.out
#SBATCH --error=logs/regen-%j.err

set -eo pipefail
mkdir -p logs

module load miniconda
set +u
source /apps/software/system/software/miniconda/24.11.3/etc/profile.d/conda.sh
conda activate $HOME/project_pi_skr2/mcn26/capiti/.envs/esmfold
set -u

echo "=== $(date): regen downstream on $(hostname) ==="

# 1. negatives for each target
for i in 1 2 3 4 5 6 7 8 9; do
    echo "--- generate_negatives T${i} ---"
    python -m src.data.generate_negatives --target "T${i}"
done

# 2. assemble dataset
echo "--- assemble_dataset ---"
python -m src.data.assemble_dataset

# 3. train (writes to data/runs/v3/). Uses mean_max pool; the ablation
#    at docs/benchmark/v2_variants_comparison.md showed it strictly
#    dominates mean pool on every class except a small perturb30 cost.
echo "--- train ---"
python -m src.student.train --out-dir data/runs/v3 --epochs 15 \
    --pool mean_max

# 4. export ONNX
echo "--- export_onnx ---"
python -m src.student.export_onnx \
    --ckpt data/runs/v3/best.pt \
    --out-onnx data/runs/v3/capiti.onnx \
    --out-meta data/runs/v3/capiti.meta.json \
    --pool mean_max

# 5. bundle into the ab9 set (overwrite the v2 bundled model)
echo "--- bundle into capiti/_model/ab9/ ---"
cp data/runs/v3/capiti.onnx capiti/_model/ab9/capiti.onnx
cp data/runs/v3/capiti.meta.json capiti/_model/ab9/capiti.meta.json

# 6. benchmark (uses the baselines env, not esmfold, because it needs
#    blast+/sklearn/etc). --gate layers in the SIFTS-backed
#    fixed-position check on top of capiti so we can see the
#    per-class dominance vs baselines.
echo "--- benchmark ---"
set +u
conda activate $HOME/project_pi_skr2/mcn26/capiti/.envs/baselines
set -u
python -m src.eval.benchmark \
    --dataset data/dataset/dataset.tsv \
    --targets data/targets/primary_sequences \
    --capiti-onnx data/runs/v3/capiti.onnx \
    --capiti-meta data/runs/v3/capiti.meta.json \
    --capiti-name capiti_v3 \
    --out-dir data/runs/v3/benchmark \
    --gate

# copy report + plots under docs/ for the release
mkdir -p docs/benchmark/v3
cp data/runs/v3/benchmark/report.md \
   data/runs/v3/benchmark/metrics.json \
   data/runs/v3/benchmark/thresholds.json \
   data/runs/v3/benchmark/*.png \
   docs/benchmark/v3/

echo "=== $(date): done ==="
cat data/runs/v3/benchmark/report.md
