# Changelog

Research-grade. Entries are per dev session, not per release.

## Adding capiti-C and capiti-E sets

Two new bundled sets. Pipeline scaled from 9 (ab9) to 235 + 59 = 294
targets end-to-end.

- New per-set data layout at `data/<SET>/targets/...`. All downstream
  scripts parameterized by path flags or `--set` env var.
- AlphaFold support: `AF:<accession>` prefix triggers AFDB fetch
  (resolving current model version via the AFDB API) and identity
  UniProt mapping. 7 of 59 E targets are AF-only.
- `pick_best_structures.py` collapses multiple PDB candidates per
  target to one row per target (X-ray > EM > NMR, best resolution,
  chain A preferred). Resolution cap raised 3.0 -> 3.5 A.
- `max_len` bumped 800 -> 1200 everywhere to fit longer C targets.
- `train.py` / `assemble_dataset.py` now derive the target list
  dynamically — no more hardcoded `T1..T9`.
- All EBI / RCSB / UniProt / AFDB fetches now carry a user-agent and
  retry with backoff on 429 / 5xx; 100 ms polite pacing between
  requests. Needed at 200+ request scale.
- Bundle layout: `capiti/_model/{ab9,C,E}/capiti.{onnx,meta.json}`
  selectable via `--set` at CLI invocation.

### Results

| set | targets | AUC (no gate) | AUC (+gate) | ala_scan +gate |
|---|---|---|---|---|
| ab9 | 9   | 0.997 | 1.000 | 1.000 |
| E   | 59  | 0.997 | 0.984 | 0.966 |
| C   | 235 | 0.986 | 0.970 | 0.953 |

Side-by-side writeup at `docs/benchmark/CE_summary.md`. Per-set
artifacts at `docs/benchmark/{v3,E_v1,C_v1}/`.

### Bugs fixed during this scaling

- `src/eval/scorers.py` - both `load_targets()` and
  `load_gate_mask()` iterated a hardcoded `[T1..T9]` list;
  generalized to scan the directory.
- `src/data/generate_mpnn_variants.py` - chain hardcoded as A in three
  places (assign_fixed_chains, make_fixed_positions_dict, and
  `chain_a_author_range`); now respects `active_sites["chain"]`.
  Affected ~10 E targets and ~30 C targets where the chosen PDB
  chain is B/C/D/G/J/etc. Re-ran those.

### Open follow-ups

- `kmer3_lr` baseline ported to `scipy.sparse` (`_kmer_matrix_sparse`
  + sklearn `solver='liblinear'`). Memory drops ~50x at k=3, lets
  the C 318k-row baseline fit in <8 GB. C numbers now in
  `docs/benchmark/CE_summary.md`.
- 6 E + 5 C MPNN array slots originally hit the 45-minute time limit
  and were missing from the v1 dataset. Re-ran with `--time 3:00:00`
  - all 9 finished (longest 2:55). Datasets and benchmarks above
  reflect this regenerated set.

### Dropped targets (13 of 307, 4.2%)

Captured in `configs/targets-{C,E}.review.tsv`:

- **10 CIF-only PDBs** (RCSB retired legacy `.pdb` for these, often
  because the molecule is too large or has multi-char / numeric chain
  IDs): T-C92, T-C150, T-C172, T-C202, T-C217, T-C223, T-C240, T-C242,
  T-E52, T-E56. Recoverable with a `.cif -> .pdb` adapter in
  `setup_set.py`; ProteinMPNN's parser might still complain about the
  multi-char chains in some of these.
- **3 too-low-resolution**: T-C4 (EM 9.8 A), T-C194 (NMR only), T-C5
  (source list had no usable metadata). Can add by loosening the
  picker's cap per-target.

No retraining on paper here: if we want them back, add the fix and
regen the affected set.

## v3 — max pool + inference-time fixed-position gate

Ships as the `ab9` bundled model. At natural threshold (0.5),
`capiti_v3+gate` ties or beats every baseline on every class on the
test split (one row off on `perturb30`). AUC and PR-AUC both round to
1.000. Full writeup in `docs/benchmark/v3_writeup.md`.

### Architecture: max pool

`CapitiCNN` gains a `pool` kwarg. `pool="mean_max"` concatenates
masked-mean and masked-max pooling over the sequence axis and widens
the MLP head's input to `2 * channels`. Total params: 227k (was 219k
with plain mean pool).

Ablation showed max pool was the decisive architectural change for
detecting single-residue active-site knockouts (ala_scan class),
going from 31.7% to 58.5% at matched 5% FPR. An auxiliary per-residue
"is this a fixed-position" head was tried and did not stack; dropped
from the final model. See `docs/benchmark/v2_variants_comparison.md`.

### Inference-time fixed-position gate

New `src/eval/scorers.apply_fixed_position_gate` plus `--gate` flag
on `src.eval.benchmark`. At inference, use capiti's top-Ti prediction
to look up Ti's fixed-position mask from the SIFTS residue map; if the
query has a mutation at any fixed position (checked in whichever
coord system matches the query's length), zero the in-set score.

Adds no new parameters, adds negligible runtime cost (<1 ms per
sequence on CPU). Drives ala_scan to 100% at 0.5 threshold without
denting any other class.

### Pipeline bug fixes

Three new bugs in the SIFTS pipeline, found while getting the gate
working:

- `mpnn_1idx` was computed as rank among SIFTS-observed residues, but
  ProteinMPNN's `parse_multiple_chains.py` builds its sequence from
  the PDB author-number range `lo..hi` with gaps as `-`. For targets
  with internal PDB gaps (T3, T4), mpnn positions were shifted. Now
  compute as `pdb_num - lo_author + 1`.
- `lo_author` was `min(pdb_num)` over SIFTS-observed records, but
  some PDBs have leading residues SIFTS doesn't map to UniProt (T8
  has a Met at author num 0). ProteinMPNN's parser sees them. Fixed
  by taking `lo_author` from the PDB's own ATOM CA records.
- SIFTS can flag a residue "observed" when its atoms live in HETATM
  records (e.g., T3's carbamylated lysine = KCX at U73). ProteinMPNN
  only reads ATOM, so it can't preserve those. Now we detect them
  and flag as unresolved.
- `fetch_active_sites.py` was expanding UniProt disulfide-bond
  features (`unp_start=29, unp_end=66`) as a 38-residue range, but
  the biologically-fixed residues are only the two bonded cysteines.
  T8 was the affected target; narrowed to `[29, 66]` and the
  perturb30 false positives collapsed. Fix in-place in
  `src/data/fetch_active_sites.py` so future targets don't hit it.

### Other

- `ResidueMap` gained `expected_for_gate()` and `mpnn_length` field.
- `build_residue_map.py` now parses the PDB directly (via
  `pdb_chain_ca_keys`) to cross-check SIFTS's observed flag against
  what MPNN's parser will actually see.
- All sbatch scripts dropped to `--gpus=rtx_5000_ada:1` and tight
  mem / time after the resource audit in `docs/resource_audit.md`.
  Typical queue wait on priority_gpu dropped from tens-of-minutes on
  H200 to effectively immediate.
- Docs refreshed: `docs/capiti.overview.svg` shows the concat
  mean+max pool; `docs/capiti.summary.txt` reflects the 227k-param
  size with the wider head.

### Known follow-up

The gate currently lives in `src/eval/` and runs at benchmark time,
not in the shipped `capiti` CLI. To get the ala_scan=1.0 numbers in
production, we need to bundle a per-target gate JSON (triples +
reference lengths) into `capiti/_model/ab9/` and wire it into
`capiti/cli.py`. Small data (<5 KB per target set), small code
change. Next release.

## v2 — SIFTS-backed coordinate system

Triggered by an ala_scan audit that showed 91% of test rows were
mislabeled: `generate_negatives.py` applied ala substitutions at
WT-FASTA indices on top of MPNN designs, but the two coordinate systems
are not aligned. Downstream of that, `generate_mpnn_variants.py`'s
`a - lo + 1` math was wrong for all targets with PDB gaps or N-terminal
construct differences (everything except T1/T5/T9). MPNN was preserving
*some* position consistently, just not the catalytic one the mask
claimed.

Fix: route everything through **SIFTS** as the single source of truth.

### New modules

- `src/data/residue_map.py` — small loader exposing
  `wt_idx()`, `pdb_num()`, `mpnn_1idx()`, `fixed_wt_idx(list)`,
  `fixed_mpnn_1idx(list)`, `uniprot_for_pdb()`. One lookup API, no
  coordinate arithmetic anywhere else.
- `src/data/build_residue_map.py` — fetches SIFTS XML + canonical
  UniProt FASTA (cached under `data/targets/sifts/` and
  `data/targets/uniprot/`), builds per-target
  `data/targets/residue_maps/Ti.json`. Cross-checks AA letters
  between WT FASTA, PDB, and UniProt; fails loudly on mismatch.
  Stamps `fixed_positions_uniprot` into `active_sites/Ti.json`.

### Ported

- `src/data/generate_negatives.py` — reads
  `fixed_positions_uniprot` + `ResidueMap`. Uses `fixed_wt_idx` for
  WT-background mutations and `fixed_mpnn_0idx` for MPNN-background
  mutations (the core bug fix).
- `src/data/generate_mpnn_variants.py` — uses
  `ResidueMap.fixed_mpnn_1idx()` to pass MPNN correct 1-indexed fixed
  positions.
- `src/data/fetch_active_sites.py` — emits
  `fixed_positions_uniprot` natively for new targets.

### Canonical UniProt as source of truth

Active-site masks are now stored as **UniProt residue numbers** in
`active_sites/Ti.json` under `fixed_positions_uniprot`. Everything else
(WT-FASTA indices, MPNN 1-indexed positions, PDB author numbers) is
derived per-target by the residue map cache. See
`docs/adding_a_target.md` for the playbook.

### Data regeneration

- Old variants + dataset backed up under `data/variants_v1_preSIFTS/`,
  `data/dataset_v1_preSIFTS/`.
- MPNN positives regenerated for all 9 targets (correct fixed positions).
- Negatives regenerated from the corrected MPNN output.
- Dataset reassembled; student retrained; bundled model at
  `capiti/_model/ab9/` points at v2.

### Other

- `.gitignore` anchored `data/` to the repo root (`/data`). The
  unanchored `data/` pattern was silently ignoring `src/data/` — the
  entire data-generation module was untracked prior to this change.
- Small AA-letter mismatches surfaced on T4/T5/T8 between WT FASTA and
  UniProt canonical sequence (construct-specific point differences or
  engineered mutants); logged as diagnostics, not blocking.

## Benchmark pipeline

Reusable capiti-vs-baselines benchmark:

- `src/eval/scorers.py` — capiti (ONNX), BLAST nearest-WT bit score,
  k-mer NN to target seqs, k-mer logistic regression. Optional
  inference-time fixed-position gate (capiti prediction drives target
  selection; gate zeros score if any fixed position mutated).
- `src/eval/plots.py` — ROC and PR overlays, per-class bar chart,
  score-distribution violins, per-class delta vs. a reference method.
- `src/eval/benchmark.py` — CLI: scores all methods on a split, picks a
  per-method operating point at a target FPR on negatives, writes
  scores + metrics + thresholds + report + plots.

Use it on any future `--capiti-onnx` / `--capiti-meta`:

```
python -m src.eval.benchmark \
    --dataset data/dataset/dataset.tsv \
    --targets data/targets/primary_sequences \
    --capiti-onnx data/runs/<run>/capiti.onnx \
    --capiti-meta data/runs/<run>/capiti.meta.json \
    --capiti-name <label> \
    --out-dir data/runs/<run>/benchmark
```

Reports + plots for releases live under `docs/benchmark/<run>/`.

## Multiple reference sets

The CLI now selects a bundled reference set at invocation:

- `capiti <nt_seq>` defaults to `ab9`.
- `capiti --set ab9|C|E` picks a set. `CAPITI_SET` env var also works.
- `--model` / `--meta` still override with explicit paths.

Layout: `capiti/_model/<set>/capiti.{onnx,meta.json}`. `ab9` ships the
v2 model. `C` and `E` are placeholder dirs; drop model files in and the
CLI will pick them up. Package-data glob updated in `pyproject.toml`.

## Env split

Second conda env `capiti-baselines` (`configs/envs/baselines.yml`,
install path `.envs/baselines`) hosts BLAST+, scikit-learn,
onnxruntime, matplotlib, pandas. Kept separate from `capiti-fold` so
its torch/CUDA pins don't fight bioconda `blast` glibc-sensitive deps.
