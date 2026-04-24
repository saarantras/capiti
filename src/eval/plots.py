"""Benchmark plots: ROC / PR overlays, per-class bars, score distributions.

All plots take a pandas DataFrame with columns:
    id, class, binary_label, target, <method_1>, <method_2>, ...
where each <method_*> column holds that method's in-set score.

Threshold-dependent plots take a dict `thresholds[method] = float` chosen
externally (e.g. to match a target FPR on test negatives).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score, roc_curve,
)


def _methods(df):
    reserved = {"id", "class", "binary_label", "target", "split"}
    return [c for c in df.columns if c not in reserved]


def plot_roc(df, out_path):
    methods = _methods(df)
    y = df["binary_label"].to_numpy().astype(int)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for m in methods:
        fpr, tpr, _ = roc_curve(y, df[m].to_numpy())
        auc = roc_auc_score(y, df[m].to_numpy())
        ax.plot(fpr, tpr, label=f"{m}  AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.5, alpha=0.5)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC (in-set vs not)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pr(df, out_path):
    methods = _methods(df)
    y = df["binary_label"].to_numpy().astype(int)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for m in methods:
        prec, rec, _ = precision_recall_curve(y, df[m].to_numpy())
        ap = average_precision_score(y, df[m].to_numpy())
        ax.plot(rec, prec, label=f"{m}  AP={ap:.3f}")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision-Recall (in-set)")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_class_acc(df, thresholds, out_path, class_order=None):
    """Grouped bar chart: binary correctness per class per method at the
    given per-method operating point."""
    methods = _methods(df)
    classes = class_order or sorted(df["class"].unique())

    # correctness per (class, method)
    acc = np.zeros((len(classes), len(methods)))
    for i, cls in enumerate(classes):
        sub = df[df["class"] == cls]
        y = sub["binary_label"].to_numpy().astype(int)
        for j, m in enumerate(methods):
            pred = (sub[m].to_numpy() >= thresholds[m]).astype(int)
            acc[i, j] = (pred == y).mean() if len(y) else np.nan

    width = 0.8 / max(len(methods), 1)
    x = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(classes)), 4.5))
    for j, m in enumerate(methods):
        ax.bar(x + j * width - 0.4 + width / 2, acc[:, j], width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=25, ha="right")
    ax.set_ylabel("binary accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-class accuracy at matched operating point")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_score_distributions(df, out_path, class_order=None):
    """One subplot per method; per-class violin of scores."""
    methods = _methods(df)
    classes = class_order or sorted(df["class"].unique())

    fig, axes = plt.subplots(len(methods), 1,
                             figsize=(max(7, 1.0 * len(classes)),
                                      2.2 * len(methods)),
                             sharex=True)
    if len(methods) == 1:
        axes = [axes]
    for ax, m in zip(axes, methods):
        data = [df.loc[df["class"] == c, m].to_numpy() for c in classes]
        parts = ax.violinplot(data, showmedians=True, widths=0.8)
        # colour positives vs negatives by majority of binary_label per class
        for i, cls in enumerate(classes):
            sub = df[df["class"] == cls]
            pos = (sub["binary_label"].astype(int) == 1).mean() > 0.5
            color = "#2b8cbe" if pos else "#e34a33"
            parts["bodies"][i].set_facecolor(color)
            parts["bodies"][i].set_alpha(0.6)
        ax.set_ylabel(m, fontsize=9)
        ax.grid(axis="y", alpha=0.2)
    axes[-1].set_xticks(np.arange(1, len(classes) + 1))
    axes[-1].set_xticklabels(classes, rotation=25, ha="right")
    fig.suptitle("Score distributions by class (blue=positive, red=negative)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_method_delta(df, thresholds, out_path, reference, class_order=None):
    """Per-class (reference - other) binary accuracy delta; positive means
    reference method wins. Highlights where the biggest gaps are."""
    methods = [m for m in _methods(df) if m != reference]
    classes = class_order or sorted(df["class"].unique())

    ref_acc = np.zeros(len(classes))
    other_acc = np.zeros((len(classes), len(methods)))
    for i, cls in enumerate(classes):
        sub = df[df["class"] == cls]
        y = sub["binary_label"].to_numpy().astype(int)
        r = (sub[reference].to_numpy() >= thresholds[reference]).astype(int)
        ref_acc[i] = (r == y).mean() if len(y) else np.nan
        for j, m in enumerate(methods):
            p = (sub[m].to_numpy() >= thresholds[m]).astype(int)
            other_acc[i, j] = (p == y).mean() if len(y) else np.nan
    delta = ref_acc[:, None] - other_acc

    width = 0.8 / max(len(methods), 1)
    x = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(classes)), 4.5))
    for j, m in enumerate(methods):
        ax.bar(x + j * width - 0.4 + width / 2, delta[:, j], width, label=m)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=25, ha="right")
    ax.set_ylabel(f"{reference} acc  -  other acc")
    ax.set_title(f"Per-class accuracy delta vs {reference}")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
