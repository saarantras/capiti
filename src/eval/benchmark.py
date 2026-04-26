"""Benchmark CapitiCNN against BLAST and k-mer baselines.

Runs all scorers on a held-out split, picks a per-method operating point
by matching a target FPR on that split's negatives (apples-to-apples),
then writes scores, metrics, and plots to --out-dir.

Re-usable: point --capiti-onnx / --capiti-meta at any future model. The
dataset schema (id, seq, target, class, binary_label, split) is what ties
a run to its benchmark.

Example:
    python -m src.eval.benchmark \\
        --dataset data/dataset/dataset.tsv \\
        --targets data/targets/primary_sequences \\
        --capiti-onnx data/runs/v1/capiti.onnx \\
        --capiti-meta data/runs/v1/capiti.meta.json \\
        --out-dir data/runs/v1/benchmark
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.eval import plots as _plots
from src.eval import scorers as _s


# Natural order for per-class plots: positives first, then negatives from
# easiest to hardest.
CLASS_ORDER = [
    "wt", "mpnn_positive",
    "ala_scan", "combined_ko",
    "family_decoy", "perturb30", "scramble", "random_decoy",
]


def load_dataset(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def fpr_threshold(scores, y, target_fpr):
    neg = scores[y == 0]
    if len(neg) == 0:
        return float("-inf")
    return float(np.quantile(neg, 1 - target_fpr))


def summarize(scores, rows, thr):
    y = np.array([int(r["binary_label"]) for r in rows])
    pred = (scores >= thr).astype(int)
    by_cls = defaultdict(list)
    for s, r, p in zip(scores, rows, pred):
        by_cls[r["class"]].append((int(r["binary_label"]), int(p), float(s)))
    per_class = {}
    for cls, items in by_cls.items():
        yc = np.array([a for a, _, _ in items])
        pc = np.array([b for _, b, _ in items])
        sc = np.array([c for _, _, c in items])
        per_class[cls] = {
            "n": int(len(items)),
            "acc": float((pc == yc).mean()),
            "mean_score": float(sc.mean()),
            "median_score": float(np.median(sc)),
        }
    return {
        "binary_auc": float(roc_auc_score(y, scores)),
        "binary_prauc": float(average_precision_score(y, scores)),
        "threshold": thr,
        "overall_acc": float((pred == y).mean()),
        "tpr": float(pred[y == 1].mean()) if (y == 1).any() else 0.0,
        "fpr": float(pred[y == 0].mean()) if (y == 0).any() else 0.0,
        "per_class": per_class,
    }


def write_markdown_report(metrics, class_order, out_md, fpr):
    methods = list(metrics.keys())
    lines = [
        f"# Benchmark report (operating point: FPR <= {fpr})",
        "",
        "## Overall",
        "",
        "| method | AUC | PR-AUC | acc | TPR | FPR | threshold |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in methods:
        r = metrics[m]
        lines.append(
            f"| {m} | {r['binary_auc']:.3f} | {r['binary_prauc']:.3f} | "
            f"{r['overall_acc']:.3f} | {r['tpr']:.3f} | {r['fpr']:.3f} | "
            f"{r['threshold']:.4g} |"
        )
    lines += ["", "## Binary accuracy per class", ""]
    header = "| class | n | " + " | ".join(methods) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (2 + len(methods)))
    all_classes = [c for c in class_order
                   if any(c in metrics[m]["per_class"] for m in methods)]
    for cls in all_classes:
        any_n = next((metrics[m]["per_class"][cls]["n"]
                      for m in methods if cls in metrics[m]["per_class"]),
                     0)
        row = [cls, str(any_n)]
        for m in methods:
            pc = metrics[m]["per_class"].get(cls)
            row.append(f"{pc['acc']:.3f}" if pc else "-")
        lines.append("| " + " | ".join(row) + " |")
    Path(out_md).write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--targets", required=True,
                    help="dir with T1..T9.fasta")
    ap.add_argument("--capiti-onnx", required=True)
    ap.add_argument("--capiti-meta", required=True)
    ap.add_argument("--capiti-name", default="capiti",
                    help="label for this capiti run in tables/plots")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--fpr", type=float, default=0.05,
                    help="target FPR for matched operating point")
    ap.add_argument("--k", type=int, default=3, help="k-mer size")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="method names to skip, e.g. --skip kmer_lr blast")
    ap.add_argument("--gate", action="store_true",
                    help="also report a gated capiti variant that zeros "
                         "the score when the model's predicted target "
                         "has a mutated active-site residue (uses SIFTS "
                         "residue maps to check both WT and MPNN "
                         "coordinate systems)")
    ap.add_argument("--active-sites", default="data/targets/active_sites")
    ap.add_argument("--residue-maps", default="data/targets/residue_maps")
    ap.add_argument("--gate-conf", type=float, default=0.2,
                    help="min prob on predicted target for the gate to "
                         "trigger (low-conf rows passed through)")
    ap.add_argument("--prefix-frac", type=float, default=1.0,
                    help="if <1.0, truncate every eval seq to "
                         "int(frac * len(seq)) before scoring. Used by "
                         "the streaming-inference prefix sweep to "
                         "measure per-class TPR/FPR vs prefix length.")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = load_dataset(args.dataset)
    train = [r for r in rows if r["split"] == "train"]
    eval_rows = [r for r in rows if r["split"] == args.split]
    print(f"[benchmark] train={len(train)} {args.split}={len(eval_rows)}",
          file=sys.stderr)

    if args.prefix_frac < 1.0:
        # Truncate eval seqs in place for the streaming-prefix sweep.
        # Train rows are left intact: the kmer_lr classifier is fit on
        # full-length training data and queried with truncated test
        # rows, mirroring the deployed streaming setup. BLAST short
        # prefixes that miss the seed return no hit -> score 0, which
        # is what we want.
        frac = args.prefix_frac
        trunc = []
        for r in eval_rows:
            r2 = dict(r)
            L = len(r2["seq"])
            keep = max(1, int(L * frac))
            r2["seq"] = r2["seq"][:keep]
            trunc.append(r2)
        eval_rows = trunc
        print(f"[benchmark] prefix-frac={frac}: truncated {len(eval_rows)} "
              f"eval rows", file=sys.stderr)

    target_seqs = _s.load_targets(args.targets)

    score_cols = {}
    method_order = []

    if "capiti" not in args.skip:
        print(f"[benchmark] scoring {args.capiti_name} ...", file=sys.stderr)
        if args.gate:
            base, probs, labels = _s.capiti_scores(
                eval_rows, args.capiti_onnx, args.capiti_meta,
                return_probs=True)
            score_cols[args.capiti_name] = base
            method_order.append(args.capiti_name)
            print("[benchmark] applying fixed-position gate ...",
                  file=sys.stderr)
            gate_mask = _s.load_gate_mask(args.active_sites,
                                           args.residue_maps)
            preds = _s.capiti_predicted_targets(probs, labels)
            gated_name = f"{args.capiti_name}+gate"
            score_cols[gated_name] = _s.apply_fixed_position_gate(
                eval_rows, gate_mask, base, preds,
                target_conf_min=args.gate_conf)
            method_order.append(gated_name)
        else:
            score_cols[args.capiti_name] = _s.capiti_scores(
                eval_rows, args.capiti_onnx, args.capiti_meta)
            method_order.append(args.capiti_name)

    if "blast" not in args.skip:
        print("[benchmark] scoring blast_nearest_wt ...", file=sys.stderr)
        with tempfile.TemporaryDirectory() as td:
            score_cols["blast_nearest_wt"] = _s.blast_scores(
                eval_rows, target_seqs, td)
        method_order.append("blast_nearest_wt")

    if "kmer_nn" not in args.skip:
        name = f"kmer{args.k}_nn"
        print(f"[benchmark] scoring {name} ...", file=sys.stderr)
        score_cols[name] = _s.kmer_nn_scores(eval_rows, target_seqs, k=args.k)
        method_order.append(name)

    if "kmer_lr" not in args.skip:
        name = f"kmer{args.k}_lr"
        print(f"[benchmark] scoring {name} ...", file=sys.stderr)
        score_cols[name] = _s.kmer_lr_scores(train, eval_rows, k=args.k)
        method_order.append(name)

    # scores dataframe
    base = pd.DataFrame({
        "id": [r["id"] for r in eval_rows],
        "class": [r["class"] for r in eval_rows],
        "target": [r["target"] for r in eval_rows],
        "binary_label": [int(r["binary_label"]) for r in eval_rows],
    })
    for m in method_order:
        base[m] = score_cols[m]
    base.to_csv(out / "scores.tsv", sep="\t", index=False)
    print(f"[benchmark] scores -> {out / 'scores.tsv'}", file=sys.stderr)

    # matched-FPR thresholds + metrics
    y = base["binary_label"].to_numpy().astype(int)
    thresholds = {m: fpr_threshold(base[m].to_numpy(), y, args.fpr)
                  for m in method_order}
    metrics = {m: summarize(base[m].to_numpy(), eval_rows, thresholds[m])
               for m in method_order}
    metrics_out = dict(metrics)
    metrics_out["_meta"] = {"prefix_frac": args.prefix_frac,
                             "fpr_target": args.fpr}
    (out / "metrics.json").write_text(json.dumps(metrics_out, indent=2))
    (out / "thresholds.json").write_text(json.dumps(thresholds, indent=2))

    write_markdown_report(metrics, CLASS_ORDER, out / "report.md", args.fpr)

    # plots
    cls_present = [c for c in CLASS_ORDER if c in set(base["class"])]
    _plots.plot_roc(base, out / "roc.png")
    _plots.plot_pr(base, out / "pr.png")
    _plots.plot_per_class_acc(
        base, thresholds, out / "per_class_acc.png", class_order=cls_present)
    _plots.plot_score_distributions(
        base, out / "score_distributions.png", class_order=cls_present)
    _plots.plot_method_delta(
        base, thresholds, out / "delta_vs_capiti.png",
        reference=args.capiti_name, class_order=cls_present)

    print(f"[benchmark] wrote report + plots to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
