"""Assemble all variant / decoy / WT FASTAs into a single TSV dataset
with stratified train/val/test splits.

Columns:
  id              : unique per row
  seq             : amino-acid sequence (X allowed for unresolved)
  target          : T1..T9 for in-set positives, "none" otherwise
  class           : wt, mpnn_positive, ala_scan, combined_ko, scramble,
                    perturb30, family_decoy, random_decoy
  binary_label    : 1 if target != "none" else 0
  split           : train / val / test (stratified by target*class)

Writes: data/dataset/dataset.tsv + data/dataset/summary.json
"""
import argparse, csv, json, pathlib, random, sys


def read_fasta(p):
    name, seq = None, []
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                yield name, "".join(seq)
            name = line[1:].split()[0]
            seq = []
        else:
            seq.append(line)
    if name is not None:
        yield name, "".join(seq)


TARGETS = [f"T{i}" for i in range(1, 10)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/dataset")
    ap.add_argument("--primary-seqs", default="data/targets/primary_sequences")
    ap.add_argument("--variants", default="data/variants")
    ap.add_argument("--decoys", default="data/decoys")
    ap.add_argument("--min-len", type=int, default=30)
    ap.add_argument("--max-len", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.10)
    args = ap.parse_args()

    random.seed(args.seed)
    rows = []

    def add(sid, seq, target, cls):
        seq = seq.replace("*", "")
        if len(seq) < args.min_len or len(seq) > args.max_len:
            return
        rows.append(dict(id=sid, seq=seq, target=target, cls=cls,
                         binary_label=1 if target != "none" else 0))

    # WT sequences (one per target)
    for t in TARGETS:
        p = pathlib.Path(args.primary_seqs, f"{t}.fasta")
        if not p.exists():
            continue
        for name, seq in read_fasta(p):
            add(f"{t}_wt", seq, t, "wt")

    # MPNN positives -> target t, binary=1
    for t in TARGETS:
        p = pathlib.Path(args.variants, "mpnn_positives", f"{t}.fasta")
        if not p.exists():
            continue
        for name, seq in read_fasta(p):
            add(name, seq, t, "mpnn_positive")

    # Variant-based negatives (ala_scan, combined_ko, scramble, perturb30)
    for cls in ("ala_scan", "combined_ko", "scramble", "perturb30"):
        for t in TARGETS:
            p = pathlib.Path(args.variants, cls, f"{t}.fasta")
            if not p.exists():
                continue
            for name, seq in read_fasta(p):
                add(name, seq, "none", cls)

    # Family decoys (per target, labelled none)
    for t in TARGETS:
        p = pathlib.Path(args.decoys, "family", f"{t}.fasta")
        if not p.exists():
            continue
        for name, seq in read_fasta(p):
            add(f"fam_{t}_{name}", seq, "none", "family_decoy")

    # Random background decoys
    p = pathlib.Path(args.decoys, "random.fasta")
    if p.exists():
        for name, seq in read_fasta(p):
            add(f"rand_{name}", seq, "none", "random_decoy")

    # Dedupe by seq (keep first)
    seen = set(); deduped = []
    for r in rows:
        if r["seq"] in seen:
            continue
        seen.add(r["seq"])
        deduped.append(r)
    rows = deduped

    # Stratified split by (target, cls)
    buckets = {}
    for r in rows:
        buckets.setdefault((r["target"], r["cls"]), []).append(r)
    for key, rs in buckets.items():
        random.shuffle(rs)
        n = len(rs)
        n_test = max(1, int(args.test_frac * n)) if n >= 10 else 0
        n_val = max(1, int(args.val_frac * n)) if n >= 10 else 0
        for i, r in enumerate(rs):
            if i < n_test:
                r["split"] = "test"
            elif i < n_test + n_val:
                r["split"] = "val"
            else:
                r["split"] = "train"

    out = pathlib.Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    with open(out / "dataset.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["id", "seq", "target", "class", "binary_label", "split"])
        for r in rows:
            w.writerow([r["id"], r["seq"], r["target"], r["cls"],
                        r["binary_label"], r["split"]])

    # Summary
    summary = dict(total=len(rows), by_split={}, by_class={},
                   by_target={}, by_cell={})
    for r in rows:
        summary["by_split"][r["split"]] = summary["by_split"].get(r["split"], 0) + 1
        summary["by_class"][r["cls"]] = summary["by_class"].get(r["cls"], 0) + 1
        summary["by_target"][r["target"]] = summary["by_target"].get(r["target"], 0) + 1
        cell = f"{r['target']}/{r['cls']}/{r['split']}"
        summary["by_cell"][cell] = summary["by_cell"].get(cell, 0) + 1
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {len(rows)} rows -> {out/'dataset.tsv'}")
    print("by_split:", summary["by_split"])
    print("by_class:", summary["by_class"])


if __name__ == "__main__":
    sys.exit(main())
