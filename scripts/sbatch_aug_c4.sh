#!/bin/bash
# c4: keep_full=0.2, min_frac=0.20, bias=uniform
# (c1's free-lunch + c2's short-prefix exposure, without c3's
# biased-short skew)
#SBATCH --job-name=capiti-c4
#SBATCH --partition=priority_gpu
#SBATCH --account=prio_skr2
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=logs/aug-c4-%j.out
#SBATCH --error=logs/aug-c4-%j.err

set -eo pipefail
mkdir -p logs

OUT="data/runs/v4_aug_c4"

module load miniconda
set +u
source /apps/software/system/software/miniconda/24.11.3/etc/profile.d/conda.sh
conda activate $HOME/project_pi_skr2/mcn26/capiti/.envs/esmfold
set -u

python -m src.student.train \
    --dataset data/dataset/dataset.tsv \
    --out-dir "${OUT}" \
    --epochs 25 \
    --pool mean_max \
    --residue-maps data/targets/residue_maps \
    --active-sites data/targets/active_sites \
    --prefix-aug \
    --prefix-keep-full 0.2 \
    --prefix-min-frac 0.20 \
    --prefix-bias uniform

python -m src.student.export_onnx \
    --ckpt "${OUT}/best.pt" \
    --out-onnx "${OUT}/capiti.onnx" \
    --out-meta "${OUT}/capiti.meta.json" \
    --pool mean_max

set +u
conda activate $HOME/project_pi_skr2/mcn26/capiti/.envs/baselines
set -u

python scripts/run_prefix_sweep.py \
    --dataset data/dataset/dataset.tsv \
    --targets data/targets/primary_sequences \
    --capiti-onnx "${OUT}/capiti.onnx" \
    --capiti-meta "${OUT}/capiti.meta.json" \
    --capiti-name capiti_ab9_c4 \
    --active-sites data/targets/active_sites \
    --residue-maps data/targets/residue_maps \
    --out-dir "${OUT}/prefix_sweep"

# extend the comparison table
python scripts/compare_aug_configs.py \
    --runs-root data/runs \
    --configs c0_baseline:0.5:0.40:uniform c1_morepartial:0.2:0.40:uniform \
              c2_shorter:0.5:0.20:uniform c3_aggressive:0.2:0.20:short \
              c4:0.2:0.20:uniform \
    --out data/runs/v4_augsweep_comparison.tsv

cat data/runs/v4_augsweep_comparison.tsv
