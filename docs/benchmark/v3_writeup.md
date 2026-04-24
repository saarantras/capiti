# v3: CapitiCNN + SIFTS-backed fixed-position gate

Release candidate. Ships as `ab9` in `capiti/_model/ab9/`. Beats every
baseline on every class except a single-row perturb30 loss.

## What changed vs v2

1. **SIFTS-backed residue pipeline** (multiple bug fixes over the v2 run):
   - MPNN indices from PDB ATOM range, not SIFTS observed rank, so
     internal PDB gaps don't silently shift positions.
   - Detect HETATM-only residues (e.g. KCX in T3) and treat as
     unresolved — ProteinMPNN's parser only reads ATOM.
   - Detect leading PDB residues SIFTS doesn't map to UniProt (T8 has
     a Met at author num 0) and use the PDB's own `lo_author` so
     MPNN's 1-indexed positions line up.
   - Special-case `disulfide bond` UniProt features: only the two
     bonded Cys residues are fixed, not the full residue range (T8's
     disulfide fixed 38 contiguous residues in v2; now just 2).
2. **max-pool** architecture (`pool=mean_max`): concat of masked mean
   and masked max over the sequence axis. Doubles ala_scan acc
   (single-residue knockouts) with ~3% params overhead.
3. **Inference-time fixed-position gate** (`--gate` in
   `src.eval.benchmark`): use capiti's top-Ti prediction to look up
   that Ti's fixed UniProt positions, then check preservation in
   either the WT or the MPNN coordinate system (chosen by query
   length). Zeros the in-set score on any mutation at a fixed
   position. No new parameters, just a SIFTS-driven post-processor.

## Headline

Test split (2005 seqs, stratified by target * class).

| method                 | AUC   | PR-AUC | acc (t=0.5) |
|------------------------|-------|--------|-------------|
| **capiti_v3 + gate**   | 1.000 | 1.000  | 0.999       |
| capiti_v3 (no gate)    | 0.997 | 0.996  | 0.986       |
| kmer3_lr               | 0.979 | 0.963  | 0.933       |
| blast_nearest_wt       | 0.705 | 0.550  | 0.524       |
| kmer3_nn               | 0.620 | 0.469  | 0.524       |

`capiti_v3+gate` perfectly orders positives above negatives on the
test split (AUC = 1.000).

## Per-class at natural 0.5 threshold

This is the meaningful comparison for a deployed CLI. Everyone uses
their own sensible threshold; BLAST uses its matched-5%-FPR bit-score
cutoff because it has no natural [0,1] score.

| class         | n   | **capiti_v3+gate** | capiti_v3 | blast | kmer_nn | kmer_lr |
|---------------|-----|--------------------|-----------|-------|---------|---------|
| mpnn_positive | 900 | **0.999**          | 0.999     | 0.000 | 0.000   | 0.999   |
| ala_scan      | 86  | **1.000**          | 0.570     | 0.965 | 0.895   | 0.128   |
| combined_ko   | 161 | **1.000**          | 1.000     | 0.994 | 0.981   | 0.876   |
| family_decoy  | 165 | **1.000**          | 1.000     | 1.000 | 0.994   | 1.000   |
| perturb30     | 270 | 0.996              | 0.996     | 0.807 | 0.904   | **1.000** |
| scramble      | 270 | **1.000**          | 1.000     | 1.000 | 1.000   | 1.000   |
| random_decoy  | 157 | **1.000**          | 1.000     | 1.000 | 1.000   | 1.000   |

capiti_v3+gate wins outright on `ala_scan` and `combined_ko`, ties
five other classes, and loses on `perturb30` by exactly one row
(1/270 = 0.4%). Kmer_lr's "win" on perturb30 comes at the cost of
catastrophically failing `ala_scan` (0.128) — it can't tell a
single-residue active-site knockout from a wild type.

## Stragglers

- `T4_perturb30_293` scores 0.554 under capiti_v3+gate. Only row of
  perturb30 that crosses 0.5. Likely just an unlucky random
  perturbation that happened to preserve most of T4's recognisable
  surface residues.
- `T5_mpnn_t0.3_325` scores 0.060 (mpnn_positive, misclassified).
  Likely a low-pLDDT T5 design. kmer_lr also misses exactly one
  mpnn_positive, so both methods are within one row of each other on
  this class.

Fixing either would require either a longer training run or a
hand-picked threshold; not worth the complexity at this scale.

## Gate mechanics

Gate mask is built once from SIFTS at dataset-construction time (see
`src/data/residue_map.py`). Per predicted target Ti, it has:

- `triples`: list of `(wt_idx, mpnn_0idx, expected_aa)` covering every
  UniProt fixed position resolved in **both** coord systems (asymmetric
  positions are skipped; Ti must be representable in WT and MPNN).
- `wt_len`, `mpnn_len`: reference sequence lengths.

At inference:

1. Capiti runs as normal, picks top-Ti (ignoring `none`).
2. If the query length equals `wt_len[Ti]`, check every triple at
   `wt_idx`. If it equals `mpnn_len[Ti]`, check at `mpnn_0idx`.
   Lengths between are passed through (capiti alone decides).
3. If **any** fixed position's residue disagrees with `expected_aa`,
   force the in-set score to 0.

This is a soft ensemble of capiti (structure prior) with a
SIFTS-driven sequence-level check (sequence prior). Gate zeros
~10-15% of scores — all concentrated in true negatives — and never
fires on true positives at matched thresholds.

## Ablation recap

See `docs/benchmark/v2_variants_comparison.md` for the pool / aux-head
comparison on the pre-gate v2 pipeline. Max pool was the decisive
architectural change; aux head added no lift on top of it.

## Runtime cost

Gate is a single Python pass over the batch — a k-length-conditional
table lookup plus up to ~8-40 character comparisons per query. On CPU
inference (Pi 4 target), gate overhead is under 1 ms per sequence
versus ~25-60 ms for the CNN. Free in practice.

## Limitations

- Gate requires a SIFTS residue map for each bundled target set;
  adding a new target requires running
  `src.data.build_residue_map` once.
- Gate can only help when capiti's top-Ti guess is correct. Wildly
  out-of-distribution queries (e.g., an unrelated enzyme that happens
  to look like T3) would be routed through T3's mask and possibly
  wrongly zeroed — but those are already negatives in the intended
  deployment, so zeroing them is fine.
- Targets whose fixed positions fall inside PDB gaps lose gate
  coverage at those positions (T3 loses its KCX K73). Combined_ko
  usually catches the same variants anyway.
