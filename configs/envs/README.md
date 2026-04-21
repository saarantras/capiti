# Environments

## esmfold

Folding oracle. Uses HuggingFace `transformers` ESMFold (`facebook/esmfold_v1`),
which sidesteps the openfold CUDA-extension install that `fair-esm[esmfold]`
requires. Weights (~3 GB) are downloaded on first `from_pretrained` call.

Install (run on a login node or CPU allocation with internet):

```
module load miniconda
mamba env create -f configs/envs/esmfold.yml \
    -p $HOME/project_pi_skr2/mcn26/capiti/.envs/esmfold
```

Activate:

```
module load miniconda
conda activate $HOME/project_pi_skr2/mcn26/capiti/.envs/esmfold
```

CUDA: pinned to 12.1 (pytorch-cuda=12.1). Compatible with H200, B200, RTX 5000
Ada, RTX Pro 6000 Blackwell partitions on Bouchet.

HF weights cache: set `HF_HOME` to a scratch or project location to keep them
off the home quota. Suggested: `export HF_HOME=$HOME/project_pi_skr2/mcn26/capiti/.cache/hf`
