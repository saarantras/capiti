# capiti

Tiny protein-function classifier for edge deployment. Given a nucleotide
sequence encoding a protein, capiti flags whether the encoded protein is
expected to retain the enzymatic function of one of a small reference
set.

Weighs ~1 MB on disk, runs inference in tens of milliseconds on a
Raspberry Pi. Trained by distilling ProteinMPNN's function-preserving
design prior into a small 1D CNN.

**Overview** (each ResidualDilatedBlock collapsed to one box):

![CapitiCNN overview](docs/capiti.overview.svg)

**Inside one ResidualDilatedBlock:**

![ResidualDilatedBlock detail](docs/capiti.block.svg)

See [`docs/capiti.summary.txt`](docs/capiti.summary.txt) for the full
per-layer size / FLOP table.

## Install

```
pip install capiti
```

## Use

```
capiti ATGCGTAAAGTGGCC...           # prints TRUE or FALSE (default set ab9)
capiti ATGCGT...  --cutoff 0.8 -v   # TRUE  p_inset=0.995
capiti --fasta seqs.fa              # batch over a FASTA
echo ATGCGT... | capiti --stdin
```

### Reference sets

`capiti` ships three bundled reference sets, selectable at invocation
time via `--set NAME` (or `CAPITI_SET`).

| set | targets | description |
|---|---|---|
| `ab9` | 9   | Beta-lactamases relevant to antibiotic resistance plus other soluble enzymes. Default. |
| `E`   | 59  | Larger enzyme panel (54 PDB + 5 AlphaFold-only entries). |
| `C`   | 235 | Broad enzyme panel sourced from PDB. |

```
capiti ATGCGT... --set ab9
capiti --fasta seqs.fa --set C
CAPITI_SET=E capiti --stdin
```

### Inference-time gate

Capiti pairs the CNN with a SIFTS-backed fixed-position gate by
default: if the model picks a target Ti and the query has a mutated
residue at any of Ti's catalytic / active-site positions, the in-set
score is forced to 0. This catches single-residue active-site
knockouts the masked-mean CNN under-weights. Disable with `--no-gate`.

Exit code is 0 on TRUE, 1 on FALSE, suitable for shell pipelines:

```
capiti ATGCGT... && echo "in set" || echo "not in set"
```

## Benchmarks

On the held-out test split for each set (gate on, natural threshold):

| set | targets | AUC | mpnn_pos | ala_scan |
|---|---|---|---|---|
| ab9 | 9   | 1.000 | 0.999 | 1.000 |
| E   | 59  | 0.984 | 0.983 | 0.966 |
| C   | 235 | 0.970 | 0.964 | 0.953 |

Side-by-side comparison with BLAST and k-mer baselines at
[`docs/benchmark/CE_summary.md`](docs/benchmark/CE_summary.md). Per-set
ROC, PR, per-class plots at [`docs/benchmark/v3/`](docs/benchmark/v3/),
[`docs/benchmark/E_v1/`](docs/benchmark/E_v1/),
[`docs/benchmark/C_v1/`](docs/benchmark/C_v1/).

## Status

Research-grade. The CLI surface (flags, stdin/FASTA behaviour, exit
codes) is stable; bundled models may be retrained and updated between
0.x releases. Not for operational use.

## License

MIT.
