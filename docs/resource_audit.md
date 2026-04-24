# Slurm resource audit

Sampled from recent capiti jobs (2026-04-20 onward). Peak RSS measured
via `sacct -j <id>.batch --format=MaxRSS`.

## What each job actually used

| job type              | example | requested            | peak RAM | wall   |
|-----------------------|---------|----------------------|----------|--------|
| MPNN gen (per target) | 9427544_4 (T4, longest) | h200, 32G, 1h | 850 MB   | 27 min |
| MPNN gen (shortest)   | 9427544_9 (T9)          | h200, 32G, 1h | 751 MB   | 4 min  |
| Capiti train          | 9081680                 | h200, 32G, 30m | 2.0 GB  | 1.7 min|
| Downstream (full)     | 9427545 (neg+asm+train+bench) | h200, 32G, 45m | 3.4 GB | 1.9 min|
| Variants train+bench  | 9430890_1 (maxpool)     | h200, 32G, 20m | 3.7 GB  | 1.9 min|
| ESMFold (sample)      | 9061680                 | RTX5000, 48G, 1h | 10.5 GB | 0.6 min|

## Readings

**All MPNN/training jobs use <4 GB RAM.** The 32 GB reservation was
defensive and has no justification in the logs. One long MPNN target
(T4, 646 residues, 1000 designs) peaked at 850 MB. Capiti training on
16k samples peaks at 2 GB (with `num_workers=2`).

**None of these jobs need an H200 specifically.** MPNN used a few
hundred MB of GPU memory (H200 has 143 GB). Capiti training uses under
1 GB. An L40S or RTX 5000 Ada would run both fine; H200 allocation is
pure queue-priority waste.

**Time limits are mostly realistic.** MPNN T4 used 27 min of a 1 h
limit — keep that. Training is 90-120 s of a 30 min limit — way loose,
cut to 10 min.

**ESMFold is the only CUDA-memory-bound step.** T4 (646 residues)
peaked at 10.5 GB, and ESMFold scales ~quadratically in residue count,
so on larger future targets it can still justify an H200. Leave the
fold sbatch alone.

## Right-sized requests (applied)

| script                          | mem  | time   | gres      |
|---------------------------------|------|--------|-----------|
| `scripts/sbatch_mpnn_array.sh`  | 6G   | 45 min | gpu:1     |
| `scripts/sbatch_train.sh`       | 4G   | 10 min | gpu:1     |
| `scripts/sbatch_regen_downstream.sh` | 4G | 20 min | gpu:1 |
| `scripts/sbatch_variants.sh`    | 4G   | 10 min | gpu:1     |
| `scripts/sbatch_fold_test.sh`   | unchanged (needs big VRAM on big targets) |

`gpu:1` (generic) lets slurm schedule onto whichever card is free; the
bouchet priority_gpu pool has more L40S / RTX 5000 than H200, so we
should land earlier in the queue.

Headroom rationale:
- mem 6G on MPNN vs 1.8G peak: 3x margin, covers larger future targets.
- time 45 min on MPNN vs 27 min peak: 1.7x; a 1000-residue target would
  still comfortably fit.
- mem 4G on training vs 3.7G peak (variants): 1.1x. The variants run is
  tight because it loads the full dataset + trains + runs the
  benchmark (BLAST subprocess + sklearn LR) all in one process. If
  variant jobs ever OOM, bump to 6G. They've got no margin at 4G.

Actually let me bump variants to 8G — that loads the dataset and
spawns BLAST / LR, and the 3.7 GB peak was at 16k-row scale; future
datasets with more targets will be larger. See applied script.

## Further optimization (not done)

1. **Run MPNN on CPU.** ProteinMPNN is small enough that sampling 500
   seqs/temp on an AMD EPYC node might take 10-30 min, similar to GPU.
   Frees all GPU time for ESMFold. Worth timing on a test run.
2. **Run training on CPU.** 15 epochs, 16k samples, 500k-param model:
   3-5 min on a modern CPU. If the queue gets tight, this is a free
   escape hatch.
3. **Array job GPU sharing.** If we ever want to run 20 MPNN jobs in
   parallel, 20 GPUs is wasteful; MIG / MPS would let 3-5 share one
   H200. Practical issue: slurm allocations are per-GPU, and the
   Bouchet priority_gpu queue doesn't expose MIG slices. Skip.
