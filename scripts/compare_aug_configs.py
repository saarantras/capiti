"""Aggregate per-target min-prefix tables across prefix-aug configs into
a single comparison TSV: rows = target, cols = config_name.

Reads each config's per_target_min_prefix.tsv from
data/runs/v4_aug_<name>/prefix_sweep/."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--configs", nargs="+", required=True,
                    help="entries shaped name:keep_full:min_frac:bias")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = Path(args.runs_root)
    frames = []
    for entry in args.configs:
        name = entry.split(":", 1)[0]
        path = base / f"v4_aug_{name}" / "prefix_sweep" / \
            "per_target_min_prefix.tsv"
        df = pd.read_csv(path, sep="\t")
        df = df[df["method"].str.startswith("capiti")][["target",
                                                         "min_prefix_for_tpr"]]
        df = df.rename(columns={"min_prefix_for_tpr": name})
        frames.append(df.set_index("target"))
    out = pd.concat(frames, axis=1).reset_index()
    out.to_csv(args.out, sep="\t", index=False, na_rep="NA")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
