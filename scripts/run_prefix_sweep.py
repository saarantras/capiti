"""Streaming-inference prefix sweep.

Re-runs `python -m src.eval.benchmark --prefix-frac F` for F in a grid,
aggregates per-row scores across the sweep, and emits:

  - prefix_scores.tsv: long-form (id, class, target, binary_label,
    prefix_frac, <method>...) joined across all fracs
  - prefix_metrics.tsv: per-method per-frac overall AUC, TPR @ matched
    FPR budget, FPR
  - per_target_min_prefix.tsv: smallest frac at which each (method,
    target) reaches >= --tpr-target TPR while overall FPR <= budget
  - prefix_curves.png: three subplots (one per method), TPR vs
    prefix_frac, one line per target
  - active_site_scatter.png: capiti-only, min-prefix-for-TPR-target vs
    min(active-site index) / wt_length

Doesn't retrain. Just sweeps eval. Usage:

    python scripts/run_prefix_sweep.py \\
        --dataset data/dataset/dataset.tsv \\
        --targets data/targets/primary_sequences \\
        --capiti-onnx data/runs/v3/capiti.onnx \\
        --capiti-meta data/runs/v3/capiti.meta.json \\
        --active-sites data/targets/active_sites \\
        --residue-maps data/targets/residue_maps \\
        --out-dir data/runs/v3/prefix_sweep
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_FRACS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60,
                 0.70, 0.80, 0.90, 1.00]


def run_one(frac, args, out_root):
    out = out_root / f"frac_{frac:.2f}"
    if (out / "scores.tsv").exists() and not args.force:
        print(f"[sweep] frac={frac}: cached", file=sys.stderr)
        return out
    cmd = [
        sys.executable, "-m", "src.eval.benchmark",
        "--dataset", args.dataset,
        "--targets", args.targets,
        "--capiti-onnx", args.capiti_onnx,
        "--capiti-meta", args.capiti_meta,
        "--out-dir", str(out),
        "--split", args.split,
        "--fpr", str(args.fpr),
        "--prefix-frac", str(frac),
        "--skip", "kmer_nn",  # kmer_lr is the stronger of the two
    ]
    print(f"[sweep] frac={frac}: running benchmark ...", file=sys.stderr)
    subprocess.run(cmd, check=True)
    return out


def aggregate_scores(out_root, fracs, dataset_path=None):
    frames = []
    for f in fracs:
        sub = out_root / f"frac_{f:.2f}" / "scores.tsv"
        df = pd.read_csv(sub, sep="\t")
        df["prefix_frac"] = f
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    if dataset_path is not None:
        # Derive absolute prefix length K = floor(frac * full_seq_len).
        # The streaming wrapper sees K (count of residues made), not
        # frac (depends on unknown total length), so K is the right
        # x-axis for any inference-time threshold schedule.
        ds = pd.read_csv(dataset_path, sep="\t",
                         usecols=["id", "seq"])
        ds["seq_len"] = ds["seq"].str.len()
        out = out.merge(ds[["id", "seq_len"]], on="id", how="left")
        out["K"] = (out["prefix_frac"] * out["seq_len"]).astype(int)
    return out


def methods_in(df):
    reserved = {"id", "class", "binary_label", "target", "split",
                "prefix_frac", "seq_len", "K", "K_bin"}
    return [c for c in df.columns if c not in reserved]


def matched_fpr_threshold(scores, y, target_fpr):
    neg = scores[y == 0]
    if len(neg) == 0:
        return float("-inf")
    return float(np.quantile(neg, 1 - target_fpr))


def per_method_per_frac(df, fpr_target):
    """For each (method, frac): pick threshold matching `fpr_target` on
    that frac's negatives, then compute overall TPR/FPR plus per-target
    TPR. Returns a long-form DataFrame."""
    records = []
    for m in methods_in(df):
        for f, sub in df.groupby("prefix_frac"):
            y = sub["binary_label"].to_numpy().astype(int)
            s = sub[m].to_numpy()
            thr = matched_fpr_threshold(s, y, fpr_target)
            pred = (s >= thr).astype(int)
            overall_tpr = pred[y == 1].mean() if (y == 1).any() else 0.0
            overall_fpr = pred[y == 0].mean() if (y == 0).any() else 0.0
            base = {"method": m, "prefix_frac": float(f),
                    "threshold": thr,
                    "overall_tpr": float(overall_tpr),
                    "overall_fpr": float(overall_fpr)}
            # per-target TPR (positives only)
            pos = sub[sub["binary_label"] == 1]
            for t, gp in pos.groupby("target"):
                p = (gp[m].to_numpy() >= thr).astype(int)
                base[f"tpr__{t}"] = float(p.mean()) if len(gp) else float("nan")
            records.append(base)
    return pd.DataFrame.from_records(records)


def per_method_per_K_bin(df, fpr_target, bin_edges):
    """Per-(method, absolute-K-bin) version of per_method_per_frac.

    Each bin gets its own threshold matched to fpr_target on that bin's
    negatives. The resulting (K_lo, K_hi, threshold) schedule is what
    the streaming wrapper would consume at inference: it knows K (the
    count of residues so far) but not the unknown total length, so a
    K-keyed schedule is the deployable form.
    """
    records = []
    K = df["K"].to_numpy()
    df = df.copy()
    df["K_bin"] = np.digitize(K, bin_edges) - 1
    df["K_bin"] = df["K_bin"].clip(0, len(bin_edges) - 2)
    for m in methods_in(df):
        for b, sub in df.groupby("K_bin"):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            y = sub["binary_label"].to_numpy().astype(int)
            s = sub[m].to_numpy()
            thr = matched_fpr_threshold(s, y, fpr_target)
            pred = (s >= thr).astype(int)
            base = {
                "method": m,
                "K_lo": int(lo), "K_hi": int(hi),
                "K_mid": float((lo + hi) / 2),
                "n": int(len(sub)),
                "threshold": thr,
                "overall_tpr": float(pred[y == 1].mean()) if (y == 1).any() else 0.0,
                "overall_fpr": float(pred[y == 0].mean()) if (y == 0).any() else 0.0,
            }
            pos = sub[sub["binary_label"] == 1]
            for t, gp in pos.groupby("target"):
                p = (gp[m].to_numpy() >= thr).astype(int)
                base[f"tpr__{t}"] = float(p.mean()) if len(gp) else float("nan")
            records.append(base)
    return pd.DataFrame.from_records(records).sort_values(["method", "K_lo"])


def min_K_per_target(metrics_K_df, tpr_target):
    """Per-(method, target): smallest K bin midpoint at which the
    per-target TPR reaches `tpr_target`. NaN if never reached."""
    rows = []
    targets = sorted({c[len("tpr__"):] for c in metrics_K_df.columns
                      if c.startswith("tpr__")})
    for m, sub in metrics_K_df.groupby("method"):
        sub = sub.sort_values("K_lo")
        for t in targets:
            col = f"tpr__{t}"
            if col not in sub:
                continue
            ok = sub[sub[col] >= tpr_target]
            min_K = float(ok["K_lo"].min()) if len(ok) else float("nan")
            rows.append({"method": m, "target": t, "min_K_for_tpr": min_K})
    return pd.DataFrame.from_records(rows)


def plot_K_curves(metrics_K_df, out_path):
    methods = sorted(metrics_K_df["method"].unique())
    targets = sorted({c[len("tpr__"):] for c in metrics_K_df.columns
                      if c.startswith("tpr__")})
    fig, axes = plt.subplots(1, len(methods),
                             figsize=(4.5 * len(methods), 4.5),
                             sharey=True)
    if len(methods) == 1:
        axes = [axes]
    cmap = plt.get_cmap("viridis")
    for ax, m in zip(axes, methods):
        sub = metrics_K_df[metrics_K_df["method"] == m].sort_values("K_lo")
        for i, t in enumerate(targets):
            col = f"tpr__{t}"
            if col not in sub:
                continue
            ax.plot(sub["K_mid"], sub[col],
                    color=cmap(i / max(len(targets) - 1, 1)),
                    alpha=0.7, lw=1.0,
                    label=t if len(targets) <= 12 else None)
        ax.plot(sub["K_mid"], sub["overall_tpr"], "k-", lw=2, label="overall TPR")
        ax.plot(sub["K_mid"], sub["overall_fpr"], "r--", lw=1.2, label="overall FPR")
        ax.set_xlabel("absolute prefix length K (residues)")
        ax.set_title(m)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("rate")
    if len(targets) <= 12:
        axes[-1].legend(fontsize=7, loc="lower right",
                        ncol=2 if len(targets) > 6 else 1)
    fig.suptitle("Per-target TPR vs absolute prefix length K "
                 "(matched-FPR threshold per K-bin)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def min_prefix_per_target(metrics_df, tpr_target):
    """For each (method, target): smallest prefix_frac at which the
    per-target TPR >= tpr_target. NaN if never reached."""
    rows = []
    targets = sorted({c[len("tpr__"):] for c in metrics_df.columns
                      if c.startswith("tpr__")})
    for m, sub in metrics_df.groupby("method"):
        sub = sub.sort_values("prefix_frac")
        for t in targets:
            col = f"tpr__{t}"
            if col not in sub:
                continue
            ok = sub[sub[col] >= tpr_target]
            min_frac = float(ok["prefix_frac"].min()) if len(ok) else float("nan")
            rows.append({"method": m, "target": t,
                         "min_prefix_for_tpr": min_frac})
    return pd.DataFrame.from_records(rows)


def plot_curves(metrics_df, out_path):
    """One subplot per method; per-target TPR vs prefix_frac."""
    methods = sorted(metrics_df["method"].unique())
    targets = sorted({c[len("tpr__"):] for c in metrics_df.columns
                      if c.startswith("tpr__")})
    fig, axes = plt.subplots(1, len(methods),
                             figsize=(4.5 * len(methods), 4.5),
                             sharey=True)
    if len(methods) == 1:
        axes = [axes]
    cmap = plt.get_cmap("viridis")
    for ax, m in zip(axes, methods):
        sub = metrics_df[metrics_df["method"] == m].sort_values("prefix_frac")
        for i, t in enumerate(targets):
            col = f"tpr__{t}"
            if col not in sub:
                continue
            ax.plot(sub["prefix_frac"], sub[col],
                    color=cmap(i / max(len(targets) - 1, 1)),
                    alpha=0.7, lw=1.0,
                    label=t if len(targets) <= 12 else None)
        ax.plot(sub["prefix_frac"], sub["overall_tpr"],
                color="k", lw=2.0, label="overall TPR")
        ax.plot(sub["prefix_frac"], sub["overall_fpr"],
                color="r", lw=1.2, ls="--", label="overall FPR")
        ax.set_xlabel("prefix fraction")
        ax.set_title(m)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("rate (TPR per target / overall)")
    if len(targets) <= 12:
        axes[-1].legend(fontsize=7, loc="lower right",
                        ncol=2 if len(targets) > 6 else 1)
    else:
        axes[-1].legend(["overall TPR", "overall FPR"],
                        fontsize=8, loc="lower right")
    fig.suptitle("Per-target TPR vs prefix fraction "
                 "(threshold matched to FPR budget at each frac)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def load_active_site_position(active_dir, residue_maps_dir, target):
    """Return min(fixed_position_uniprot_idx) / wt_length for one target,
    or None if not enough info."""
    asp = Path(active_dir) / f"{target}.json"
    rmp = Path(residue_maps_dir) / f"{target}.json"
    if not asp.exists() or not rmp.exists():
        return None
    active = json.loads(asp.read_text())
    fup = active.get("fixed_positions_uniprot", [])
    if not fup:
        return None
    rmap = json.loads(rmp.read_text())
    wt_len = rmap.get("wt_length", 0)
    if not wt_len:
        return None
    return min(fup) / wt_len


def plot_active_site_scatter(min_prefix_df, capiti_name, active_dir,
                              residue_maps_dir, out_path):
    sub = min_prefix_df[min_prefix_df["method"] == capiti_name].copy()
    if sub.empty:
        return
    xs, ys, labels = [], [], []
    for _, row in sub.iterrows():
        x = load_active_site_position(active_dir, residue_maps_dir,
                                       row["target"])
        if x is None:
            continue
        xs.append(x)
        ys.append(row["min_prefix_for_tpr"])
        labels.append(row["target"])
    if not xs:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    xs = np.array(xs)
    ys = np.array(ys, dtype=float)
    finite = np.isfinite(ys)
    ax.scatter(xs[finite], ys[finite], color="C0", alpha=0.7, s=20,
               label="reached TPR target")
    if (~finite).any():
        ax.scatter(xs[~finite], np.full((~finite).sum(), 1.05),
                   color="C3", marker="x", s=24,
                   label="never reached TPR target")
    ax.plot([0, 1], [0, 1], "k--", lw=0.5, alpha=0.5,
            label="y = x (active site = min prefix)")
    ax.set_xlabel("min(active-site uniprot idx) / wt length")
    ax.set_ylabel("min prefix frac to reach per-target TPR target")
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.10)
    ax.set_title(f"{capiti_name}: prefix needed vs active-site position")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--capiti-onnx", required=True)
    ap.add_argument("--capiti-meta", required=True)
    ap.add_argument("--capiti-name", default="capiti")
    ap.add_argument("--active-sites", default="data/targets/active_sites")
    ap.add_argument("--residue-maps", default="data/targets/residue_maps")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--fpr", type=float, default=0.05)
    ap.add_argument("--tpr-target", type=float, default=0.90,
                    help="per-target TPR threshold for the min-prefix table")
    ap.add_argument("--fracs", type=float, nargs="*", default=DEFAULT_FRACS)
    ap.add_argument("--force", action="store_true",
                    help="re-run benchmark even if scores.tsv exists")
    ap.add_argument("--k-bin-width", type=int, default=100,
                    help="bin width (in residues) for the absolute-K "
                         "analysis. Smaller bins = finer schedule but "
                         "noisier per-bin estimates.")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fracs = sorted(args.fracs)
    for f in fracs:
        run_one(f, args, out)

    df = aggregate_scores(out, fracs, dataset_path=args.dataset)
    df.to_csv(out / "prefix_scores.tsv", sep="\t", index=False)
    print(f"[sweep] aggregated -> {out / 'prefix_scores.tsv'} "
          f"({len(df):,} rows)", file=sys.stderr)

    metrics_df = per_method_per_frac(df, args.fpr)
    metrics_df.to_csv(out / "prefix_metrics.tsv", sep="\t", index=False)

    min_pre = min_prefix_per_target(metrics_df, args.tpr_target)
    min_pre.to_csv(out / "per_target_min_prefix.tsv", sep="\t", index=False)

    plot_curves(metrics_df, out / "prefix_curves.png")
    plot_active_site_scatter(min_pre, args.capiti_name,
                              args.active_sites, args.residue_maps,
                              out / "active_site_scatter.png")

    # Absolute-K analysis: this is the form the streaming wrapper
    # actually consumes. K bins span [0, max_seq_len] in --k-bin-width
    # steps; the resulting threshold-per-bin is the deployable schedule.
    if "K" in df.columns:
        max_K = int(df["K"].max()) + args.k_bin_width
        bin_edges = list(range(0, max_K + 1, args.k_bin_width))
        metrics_K = per_method_per_K_bin(df, args.fpr, bin_edges)
        metrics_K.to_csv(out / "prefix_metrics_by_K.tsv",
                         sep="\t", index=False)
        min_K = min_K_per_target(metrics_K, args.tpr_target)
        min_K.to_csv(out / "per_target_min_K.tsv",
                     sep="\t", index=False)
        plot_K_curves(metrics_K, out / "prefix_curves_by_K.png")
        # Compact threshold schedule (deployable form): just (method,
        # K_lo, K_hi, threshold).
        sched = metrics_K[["method", "K_lo", "K_hi", "n", "threshold"]]
        sched.to_csv(out / "threshold_schedule_by_K.tsv",
                     sep="\t", index=False)

    print(f"[sweep] wrote curves + scatter to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
