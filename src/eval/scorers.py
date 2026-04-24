"""Score functions for the benchmark: each takes dataset rows + resources
and returns a 1D numpy array of in-set scores (higher = more in-set).

Scores are not required to be calibrated in [0, 1] — AUC / PR-AUC are
threshold-free, and we pick an operating point per method by matching a
target FPR on the evaluation split's negatives.
"""

from __future__ import annotations

import os
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression


TARGETS = [f"T{i}" for i in range(1, 10)]
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA)


# ---------- target sequences ----------

def load_targets(target_dir):
    """Dict target_id -> WT AA sequence, taking the first record per file
    (skips monomer variants etc.)."""
    seqs = {}
    for t in TARGETS:
        fa = Path(target_dir) / f"{t}.fasta"
        with open(fa) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        seq = ""
        for ln in lines[1:]:
            if ln.startswith(">"):
                break
            seq += ln
        seqs[t] = seq
    return seqs


# ---------- CapitiCNN (ONNX) ----------

def capiti_scores(rows, onnx_path, meta_path, batch_size=64):
    import json
    import onnxruntime as ort

    with open(meta_path) as fh:
        meta = json.load(fh)
    max_len = meta["max_len"]
    vocab = meta["vocab"]
    pad_idx = vocab.get("pad", 0)
    x_idx = vocab.get("X", pad_idx)
    none_idx = meta["none_idx"]

    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.environ.get("CAPITI_THREADS", "4"))
    sess = ort.InferenceSession(str(onnx_path), sess_options=so,
                                 providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name

    def encode(seq):
        ids = [pad_idx] * max_len
        for i, c in enumerate(seq[:max_len]):
            ids[i] = vocab.get(c, x_idx)
        return ids

    scores = np.zeros(len(rows), dtype=np.float32)
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        x = np.asarray([encode(r["seq"]) for r in batch], dtype=np.int64)
        logits = sess.run(None, {inp_name: x})[0]
        logits -= logits.max(axis=-1, keepdims=True)
        e = np.exp(logits)
        probs = e / e.sum(axis=-1, keepdims=True)
        scores[start:start + len(batch)] = 1.0 - probs[:, none_idx]
    return scores


# ---------- BLAST nearest-WT ----------

def blast_scores(rows, target_seqs, workdir):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    db_fa = workdir / "targets.fa"
    with open(db_fa, "w") as f:
        for t, s in target_seqs.items():
            f.write(f">{t}\n{s}\n")
    subprocess.run(
        ["makeblastdb", "-in", str(db_fa), "-dbtype", "prot",
         "-out", str(workdir / "tdb")],
        check=True, stdout=subprocess.DEVNULL,
    )

    q_fa = workdir / "queries.fa"
    with open(q_fa, "w") as f:
        for i, r in enumerate(rows):
            f.write(f">q{i}\n{r['seq']}\n")
    out = workdir / "blast.tsv"
    subprocess.run(
        ["blastp", "-query", str(q_fa), "-db", str(workdir / "tdb"),
         "-out", str(out),
         "-outfmt", "6 qseqid sseqid bitscore evalue pident length",
         "-evalue", "10", "-max_target_seqs", "9",
         "-num_threads", str(os.cpu_count() or 4)],
        check=True,
    )
    best = np.zeros(len(rows), dtype=np.float32)
    with open(out) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            qi = int(parts[0][1:])
            bs = float(parts[2])
            if bs > best[qi]:
                best[qi] = bs
    return best


# ---------- k-mer features ----------

def build_kmer_index(k):
    kmers = [""]
    for _ in range(k):
        kmers = [p + a for p in kmers for a in AA]
    return {km: i for i, km in enumerate(kmers)}


def _kmer_vec(seq, k, index):
    v = np.zeros(len(index), dtype=np.float32)
    s = "".join(ch for ch in seq if ch in AA_SET)
    for i in range(len(s) - k + 1):
        j = index.get(s[i:i + k])
        if j is not None:
            v[j] += 1
    return v


def _kmer_matrix(rows, k, index):
    X = np.zeros((len(rows), len(index)), dtype=np.float32)
    for i, r in enumerate(rows):
        X[i] = _kmer_vec(r["seq"], k, index)
    return X


def _l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1
    return X / n


def kmer_nn_scores(rows, target_seqs, k=3):
    index = build_kmer_index(k)
    T = np.stack([_kmer_vec(s, k, index) for s in target_seqs.values()])
    X = _l2(_kmer_matrix(rows, k, index))
    T = _l2(T)
    return (X @ T.T).max(axis=1)


def kmer_lr_scores(train_rows, test_rows, k=3):
    index = build_kmer_index(k)
    Xtr = _l2(_kmer_matrix(train_rows, k, index))
    ytr = np.array([int(r["binary_label"]) for r in train_rows])
    Xte = _l2(_kmer_matrix(test_rows, k, index))
    clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]
