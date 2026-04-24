# ala_scan ablation: pool mode + aux head

Operating point: FPR <= 0.05 on test negatives (per-method).
Training: fresh retrain on the post-SIFTS dataset, 15 epochs each.
Variants differ from the v2 baseline in two knobs on `CapitiCNN`:

- `pool` = `mean` (baseline) vs `mean_max` (concat of masked mean and
  masked max over the sequence axis)
- `aux-weight` = 0 (baseline) vs 0.3 (per-residue BCE loss on "is this
  a fixed-position residue?" using the SIFTS-backed mask)

## Overall

| variant                  | AUC   | PR-AUC | acc   | TPR   |
|--------------------------|-------|--------|-------|-------|
| v2 baseline (mean, aux0) | 0.983 | 0.971  | 0.970 | 0.994 |
| v2_auxhead               | 0.987 | 0.977  | 0.970 | 0.996 |
| v2_maxpool               | **0.998** | **0.997** | **0.972** | **0.999** |
| v2_maxpool_auxhead       | 0.998 | 0.997  | 0.971 | 0.998 |

## Binary accuracy per class

| class         | n   | baseline | auxhead | maxpool   | both      |
|---------------|-----|----------|---------|-----------|-----------|
| mpnn_positive | 900 | 0.994    | 0.996   | **0.999** | 0.998     |
| **ala_scan**  | 82  | 0.317    | 0.317   | **0.585** | 0.573     |
| combined_ko   | 161 | 1.000    | 1.000   | 1.000     | 1.000     |
| family_decoy  | 165 | 1.000    | 1.000   | 1.000     | 1.000     |
| perturb30     | 270 | 1.000    | 1.000   | 0.919     | 0.922     |
| scramble      | 270 | 1.000    | 1.000   | 1.000     | 1.000     |
| random_decoy  | 157 | 1.000    | 1.000   | 1.000     | 1.000     |

## Read

- **Max pool nearly doubles ala_scan accuracy** (31.7% -> 58.5%).
  Masked mean pool was the bottleneck: a 1-residue active-site
  knockout contributes ~1/L to the pooled vector, which the main head
  struggles to resolve. Max pool preserves the strongest localized
  signal anywhere in the sequence, and that's exactly where the CNN
  lights up on an active-site change.
- **The aux head alone doesn't move ala_scan.** The per-residue
  supervision tells the trunk which positions are catalytic, but
  without max pool the head still averages the resulting features
  away. Small AUC bump (0.983 -> 0.987), no per-class win.
- **Stacking both is a wash vs max pool alone** (58.5% vs 57.3% on
  ala_scan). The aux head isn't hurting, but it's redundant once the
  pooling bottleneck is removed.
- **Max pool has a small perturb30 regression** (100% -> 92%). 30%
  random-residue perturbation occasionally produces a strong local
  activation that max pool picks up as "in-set". Net trade: +27
  correct on ala_scan (+33 point gain across 82 examples) vs -22
  correct on perturb30 (-8 point drop across 270 examples). Still a
  clear win overall, and ala_scan is the load-bearing case for
  "function broken".

## Decision

Ship `v2_maxpool` as `ab9`. It strictly dominates the baseline on AUC
(0.998 vs 0.983), PR-AUC (0.997 vs 0.971), mpnn_positive (0.999 vs
0.994), and the headline ala_scan number (0.585 vs 0.317). The
perturb30 cost is small and acceptable.

Drop the aux head from v3 — it doesn't pull its weight once max pool
is in. Revisit it if we start seeing ala_scan plateau below what a
human could get from inspection alone.

## Next levers (unused)

If we want to push ala_scan further:

- Larger `aux-weight` (tried 0.3 only). Might help once it's paired
  with max pool, but combined at 0.3 didn't beat max pool alone.
- Richer ala_scan training data: {G, D, L, charge-flip} substitutions
  at each fixed position, not just A. 5x more training rows, each
  hitting a real catalytic residue.
- Contrastive loss pushing ala_scan embeddings away from their source
  mpnn_positive's embedding. Most principled; biggest lift to
  implement.
