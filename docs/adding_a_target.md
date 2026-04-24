# Adding a new target

Everything downstream of active-site masking is driven by a single
per-target residue map (SIFTS-backed) that pins together UniProt, WT
FASTA, PDB author numbering, and ProteinMPNN's 1-indexed positions.
Adding a target is mechanical once you have a UniProt accession and a
PDB entry.

## Inputs you provide

1. A line in `configs/targets.tsv` — `target_id pdb_id chain`.
2. The PDB file at `data/targets/structures/<Ti>.pdb`.
3. A WT FASTA at `data/targets/primary_sequences/<Ti>.fasta`. This can
   be the UniProt canonical sequence (easiest), or any coherent variant
   thereof (e.g., a construct-specific sequence); the builder aligns it
   to UniProt and handles offsets. Sequences that disagree with UniProt
   at individual residues are allowed but will show up as diagnostics.
4. A UniProt accession for the target. You can start with
   `fetch_active_sites.py` (below), which fills it in from SIFTS, or
   just put it in the active_sites JSON yourself.

## Pipeline

```
configs/envs/baselines  # env with network access + biopython
```

### 1. Active sites

Either run the fetcher (populates from UniProt features + SIFTS):

```
python -m src.data.fetch_active_sites --target <Ti> --pdb <PDBID>
```

...or write `data/targets/active_sites/<Ti>.json` yourself:

```json
{
  "target": "T10",
  "pdb": "XXXX",
  "chain": "A",
  "uniprot": "P12345",
  "fixed_positions_uniprot": [70, 130, 166],
  "features": [ ... ],
  "notes": ""
}
```

`fixed_positions_uniprot` is the authoritative list. `fixed_positions_pdb`
is cosmetic.

### 2. Residue map

```
python -m src.data.build_residue_map --update-active-sites
```

For each target (or just the new one with `--targets T10`):

- fetches SIFTS XML to `data/targets/sifts/<pdbid>.xml.gz`
- fetches canonical UniProt FASTA to `data/targets/uniprot/<acc>.fasta`
- aligns the WT FASTA to UniProt to find the WT-to-UniProt offset
- writes the per-residue map to `data/targets/residue_maps/<Ti>.json`
- stamps `fixed_positions_uniprot` back into the active-sites JSON

The script prints one line per target with `residues / in_wt /
observed_in_pdb / offset / fixed_uniprot`, and exits non-zero if any
AA-letter sanity check fails. Resolve those before continuing — they
usually indicate WT-FASTA / UniProt / PDB disagreements that would
quietly corrupt downstream variants.

### 3. Variants

Given a populated residue map, variant generation is numerical:

```
# MPNN positives (needs GPU, ~10-20 min/target)
sbatch scripts/sbatch_mpnn_array.sh       # array across configs/targets.tsv

# non-MPNN negatives (CPU, seconds/target)
python -m src.data.generate_negatives --target T10
```

Both scripts read `fixed_positions_uniprot` and ask the residue map for
the right coordinate system (`fixed_wt_idx` for WT backgrounds,
`fixed_mpnn_1idx` for MPNN).

### 4. Decoys, assembly, train, eval

```
python -m src.data.fetch_decoys           # if your set needs new decoys
python -m src.data.assemble_dataset       # rebuilds data/dataset/dataset.tsv
sbatch scripts/sbatch_train.sh            # or scripts/sbatch_regen_downstream.sh
python -m src.eval.benchmark ...          # see CHANGELOG / README for form
```

## Verifying a new target

Before kicking off the expensive MPNN run, sanity-check the residue
map:

```
python - <<'EOF'
from src.data.residue_map import ResidueMap
import json
t = "T10"
active = json.load(open(f"data/targets/active_sites/{t}.json"))
fup = active["fixed_positions_uniprot"]
rmap = ResidueMap.load(f"data/targets/residue_maps/{t}.json")
print("uniprot:", fup)
print("wt_idx :", rmap.fixed_wt_idx(fup))
print("mpnn_1 :", rmap.fixed_mpnn_1idx(fup))
EOF
```

Look up each index by hand against your WT FASTA to confirm it's a
catalytic residue. 30 seconds of verification here saves the hour of
compute that regenerating MPNN would otherwise waste.

## Gotchas

- The WT FASTA must be a clean substring (or a reasonable local
  alignment) of the UniProt canonical sequence. Engineered fusions
  or chimeras will not align cleanly; if `build_residue_map` can't
  find an anchor window of 15+ residues, it fails the target.
- PDB entries with internal gaps are handled, but residues in those
  gaps have `pdb_num = null` and are excluded from MPNN mappings.
  Fixed positions falling inside a gap are silently dropped
  (diagnostics log them).
- T5 was originally a homodimer; its PDB entry (`1E4E`) covers one
  monomer only. The resulting residue map is chain-A-only; fixed
  positions beyond the monomer's coverage are excluded automatically.
  New dimer / multi-chain targets behave the same way — if you need
  both chains, add them as separate targets.
- `fixed_positions_pdb` as a key in active_sites is retained for
  backward compatibility, but the generators read
  `fixed_positions_uniprot` only. If you supply a JSON without
  `fixed_positions_uniprot`, `build_residue_map` will derive one.

## What not to do

- Don't compute PDB author numbers or WT indices by hand anywhere in
  the codebase. Use `ResidueMap.*` lookups. If you find yourself
  writing `a - lo` or `a - lo + 1`, stop and open a residue map.
- Don't commit variants, datasets, or MPNN outputs into git. They're
  large and regeneratable. `data/` is gitignored at the repo root.
  Reference inputs (PDBs, active-sites JSONs, residue-map caches,
  SIFTS XMLs, UniProt FASTAs) are small and stable and are fine to
  commit if you want reproducibility without network.
