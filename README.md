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

`capiti` ships multiple bundled reference sets, selectable at invocation
time. Use `--set NAME` (or the `CAPITI_SET` env var).

| set | status | description |
|---|---|---|
| `ab9` | available | 9 soluble enzymes (several beta-lactamases relevant to antibiotic resistance, plus other enzymes). Default. |
| `C`   | planned   | TBA. |
| `E`   | planned   | TBA. |

```
capiti ATGCGT... --set ab9
capiti --fasta seqs.fa --set C
CAPITI_SET=E capiti --stdin
```

Exit code is 0 on TRUE, 1 on FALSE, suitable for shell pipelines:

```
capiti ATGCGT... && echo "in set" || echo "not in set"
```

## Status

The 0.0.x line is research-grade. The CLI surface (flags, stdin/FASTA
behaviour, exit codes) is stable; bundled models will be retrained and
updated between 0.0.x releases, and should not be used for anything
operational.

## License

MIT.
