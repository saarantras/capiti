# Environments

Two conda envs:

- `capiti-fold` (`esmfold.yml`, install path `.envs/esmfold`) — folding
  oracle, variant generation, student training. Name kept for backwards
  compatibility; "capiti" is accurate now.
- `capiti-baselines` (`baselines.yml`, install path `.envs/baselines`) —
  BLAST + k-mer baselines and the benchmark pipeline in `src/eval/`.
  Kept separate from `capiti-fold` so its torch/CUDA pins stay untouched
  by bioconda `blast` and its glibc-sensitive deps.

## capiti-fold

### What it runs

- **ESMFold oracle** (`src/oracle/`). HuggingFace `transformers`
  `EsmForProteinFolding`, which sidesteps the openfold CUDA-extension
  install that `fair-esm[esmfold]` requires. Weights (~3 GB) are
  downloaded on first `from_pretrained` call into `$HF_HOME`.
- **ProteinMPNN** (`third_party/ProteinMPNN/`, cloned separately; see
  below). Only needs PyTorch + numpy, both already in this env.
- **CapitiCNN student** (`src/student/`). Tiny model; CPU or GPU.
- **Data assembly and decoy fetch** (`src/data/`). Stdlib only for
  fetch_*; numpy for assemble_dataset.

### Install

Run on a login node or CPU allocation with internet:

```
module load miniconda
mamba env create -f configs/envs/esmfold.yml \
    -p $HOME/project_pi_skr2/mcn26/capiti/.envs/esmfold
```

Pinned versions (see `esmfold.yml`):
- `python=3.10`, `pytorch=2.3.*`, `pytorch-cuda=12.1`
- `transformers==4.46.3` (pinned; 5.x requires PyTorch >= 2.4)
- `biopython`, `numpy`, `scipy`, `tqdm`, `accelerate`, `sentencepiece`

## Activate

```
module load miniconda
set +u
source /apps/software/system/software/miniconda/24.11.3/etc/profile.d/conda.sh
conda activate $HOME/project_pi_skr2/mcn26/capiti/.envs/esmfold
set -u
```

`set +u` around activation is required: the env's MKL activate script
references `MKL_INTERFACE_LAYER` unbound, and tripping it with `set -u`
fails the shell immediately. All sbatch scripts in `scripts/` wrap the
activation in `set +u` / `set -u`.

## GPU compatibility

`pytorch-cuda=12.1` supports up to `sm_90`. That means:

| Bouchet GPU | Arch | Works? |
|---|---|---|
| RTX 5000 Ada | sm_89 | yes |
| L40S | sm_89 | yes |
| H100 / H200 | sm_90 | yes |
| B200 | sm_100 | **no** (needs PyTorch >= 2.5 + CUDA >= 12.4) |
| RTX Pro 6000 Blackwell | sm_100 | **no** (same) |

Verify with `python -c "import torch; print(torch.cuda.get_arch_list())"`
before requesting Blackwell parts.

## HF cache

Set `HF_HOME` to keep ESMFold weights off the home quota:

```
export HF_HOME=$HOME/project_pi_skr2/mcn26/capiti/.cache/hf
```

All oracle sbatch scripts already do this.

## Third-party: ProteinMPNN

ProteinMPNN is kept as a git clone outside the env (weights ship with the
repo). `third_party/` is gitignored.

```
mkdir -p third_party
git clone --depth 1 https://github.com/dauparas/ProteinMPNN third_party/ProteinMPNN
```

We use the soluble-trained weights (`soluble_model_weights/v_48_020.pt`)
since all v1 targets are soluble proteins.

## Known env quirks

- `transformers` 4.46.3's `compute_tm` hits a NaN in fp16 that crashes on
  an empty-tensor `argmax` (IndexError, "index 0 is out of bounds").
  Workaround: run ESMFold in fp32. At 261 residues the T2 fold uses
  ~9 GB VRAM on H200, well within the 143 GB card.
- ProteinMPNN's `--batch_size > --num_seq_per_target` silently emits zero
  designs. Keep `batch_size=1` (or <= num_seq_per_target).

## capiti-baselines

Drives `src/eval/` (BLAST + k-mer baselines, CapitiCNN ONNX scoring,
benchmark plots). CPU-only; no CUDA pin, no torch. Spec is
`baselines.yml`:

- `python=3.10`, `blast` (NCBI BLAST+ from bioconda), `scikit-learn`,
  `onnxruntime`, `matplotlib`, `pandas`, `numpy`, `biopython`, `tqdm`.

### Install

```
module load miniconda
mamba env create -f configs/envs/baselines.yml \
    -p $HOME/project_pi_skr2/mcn26/capiti/.envs/baselines
```

### Activate

```
module load miniconda
set +u
source /apps/software/system/software/miniconda/24.11.3/etc/profile.d/conda.sh
conda activate $HOME/project_pi_skr2/mcn26/capiti/.envs/baselines
set -u
```

### Run the benchmark

```
python -m src.eval.benchmark \
    --dataset data/dataset/dataset.tsv \
    --targets data/targets/primary_sequences \
    --capiti-onnx data/runs/v1/capiti.onnx \
    --capiti-meta data/runs/v1/capiti.meta.json \
    --out-dir data/runs/v1/benchmark
```

Writes `scores.tsv`, `metrics.json`, `thresholds.json`, `report.md`, and
ROC / PR / per-class / score-distribution / delta plots to `--out-dir`.
