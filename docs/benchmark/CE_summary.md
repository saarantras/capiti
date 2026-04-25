# capiti-C and capiti-E: scaling to many targets

Three reference sets now ship together, selectable at the CLI via
`--set <name>`:

| set | targets | UniProt-feature mask | AUC (no gate) | AUC (+gate) | dropped from list |
|---|---|---|---|---|---|
| **ab9** | 9   | yes        | 0.997 | 1.000 | 0 |
| **E**   | 59  | mostly     | 0.997 | 0.984 | 2 (CIF-only) |
| **C**   | 235 | partial    | 0.986 | 0.970 | 11 (CIF / EM / NMR) |

ab9 = the original beta-lactamase + soluble-enzyme set (v3 release).
E = 59 targets, including 7 AlphaFold-only structures (no
experimental PDB available). C = 235 targets, dominated by X-ray
structures, with several long proteins now in scope after the
`max_len 800 -> 1200` bump.

All three trained with the same architecture (`mean_max` pool,
`channels=64`, 5 dilated blocks, ~227-255k params depending on the
output head's class count). All three benchmarked with the same
SIFTS-backed inference-time gate.

## Headline

| set | AUC | PR-AUC | mpnn_pos | ala_scan | combined_ko | perturb30 |
|---|---|---|---|---|---|---|
| ab9 (gated, t=0.5)    | 1.000 | 1.000 | 0.999 | 1.000 | 1.000 | 0.948 |
| E (gated, FPR=0.05)   | 0.984 | 0.993 | 0.983 | 0.966 | 0.987 | 0.868 |
| E (no gate, FPR=0.05) | 0.997 | 0.997 | 1.000 | 0.637 | 0.999 | 0.979 |
| C (gated, FPR=0.05)   | 0.970 | 0.984 | 0.964 | 0.953 | 0.945 | 0.943 |
| C (no gate, FPR=0.05) | 0.986 | 0.986 | 0.963 | 0.460 | 0.980 | 0.999 |

Pattern is consistent across sets:

- **AUC degrades gracefully with target count.** ab9 (9) -> E (59) ->
  C (235) loses ~1.5 AUC points moving from 9 to 235 classes. This is
  the expected cost of mapping more enzymes through a fixed-capacity
  head. The model is still strongly separating positives from
  negatives at any threshold.
- **mpnn_positive recall stays high** even at 235 classes (0.97 for
  C). The student keeps the function-preserving prior intact at scale.
- **ala_scan benefits enormously from the gate** in every set: the
  CNN alone catches ~25-65% of single-residue active-site knockouts;
  the SIFTS-backed gate pushes that to 94-100% on E and C. Same
  finding as ab9, just at larger scale.
- **perturb30 cost from gate is more pronounced at scale.** ab9 had
  the gate make perturb30 slightly worse (matched-FPR threshold
  artifact); on E and C the threshold-shift cost is bigger because
  the gate zeros enough negatives to push the matched-FPR threshold
  very low (0.019 for E, 0.019 for C vs 0.005 for ab9).

## What's noticeably different vs ab9

1. **Many targets in E and C have empty `fixed_positions_uniprot`.**
   The ProteinMPNN run still preserves nothing - those targets can
   only be learned as "this fold pattern" by the CNN. The gate is a
   no-op for them. Roughly 15/59 in E and 30/235 in C have no
   UniProt feature annotations.
2. **mpnn_positive isn't 100% at 235 classes.** With 223 classes
   trying to share a 64-channel trunk, some designs from chemically
   similar targets cluster together. Mostly distinguishable but not
   perfectly. Could be addressed with wider channels or longer
   training; not yet attempted.
3. **kmer3_lr baseline ported to scipy.sparse.** The original dense
   path OOM'd on C's 318k-row x 8000-feature matrix at 48 GB. After
   switching `_kmer_matrix` to CSR sparse + sklearn's `liblinear`
   solver, it now fits comfortably. Numbers on C: AUC 0.975,
   mpnn_positive 0.828, ala_scan 0.541, perturb30 1.000. Same
   pattern as ab9: a sequence-similarity classifier loses on
   mpnn_positive and ala_scan but wins on perturb30. Capiti+gate
   beats it on every class except perturb30 (where the matched-FPR
   threshold drift costs ~6 points).
4. **3 E + 6 C MPNN array slots originally hit the 45 min time limit**
   (very large / PDB-crowded entries; longest is T-C175 at 1407
   residues). Resubmitted with `--time 3:00:00`; all 9 completed
   (longest 2:55). Datasets and benchmarks above reflect the
   regenerated set.

## Dropped vs annotated

- ab9: 0 dropped from the original 9.
- E: 2 dropped (T-E52 6G5K, T-E56 7FFE) - both CIF-only.
  Recoverable with a CIF -> PDB adapter. Net 59 / 61 trained.
- C: 11 dropped (T-C4 EM 9.8 A; T-C5 missing metadata; T-C194
  NMR-only; 8 CIF-only entries). Net 235 / 246 trained.

Loss rates: 3% on E, 4% on C. All recoverable with adapters /
hand-curation.

## Bundle layout

```
capiti/_model/
  ab9/   capiti.{onnx,meta.json}    # v3
  C/     capiti.{onnx,meta.json}    # v1
  E/     capiti.{onnx,meta.json}    # v1
```

CLI:
```
capiti ATGCGT...                  # default ab9
capiti ATGCGT... --set C
capiti ATGCGT... --set E
CAPITI_SET=C capiti --fasta x.fa
```

## Per-set artifacts

- `docs/benchmark/v3/`           - ab9 (release writeup at
                                   `docs/benchmark/v3_writeup.md`)
- `docs/benchmark/E_v1/`         - E
- `docs/benchmark/C_v1/`         - C

Each holds `report.md` + `metrics.json` + `thresholds.json` + ROC, PR,
per-class accuracy, score-distribution, and capiti-vs-other delta
plots.
