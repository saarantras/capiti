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
    """Dict target_id -> WT AA sequence, scanning every *.fasta in the
    directory and taking the first record per file. The target id is
    the file's stem, so this works for any target-naming scheme
    (ab9's `T1.fasta`, capiti-C's `T-C42.fasta`, AlphaFold targets,
    etc.) without a hardcoded list."""
    seqs = {}
    for fa in sorted(Path(target_dir).glob("*.fasta")):
        with open(fa) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        seq = ""
        for ln in lines[1:]:
            if ln.startswith(">"):
                break
            seq += ln
        if seq:
            seqs[fa.stem] = seq
    return seqs


# ---------- CapitiCNN (ONNX) ----------

def capiti_scores(rows, onnx_path, meta_path, batch_size=64,
                   return_probs=False):
    """Return in-set scores (shape N). If return_probs=True, also return
    the full per-row class probability matrix (shape N x num_labels) and
    the labels list, so callers can derive the predicted target."""
    import json
    import onnxruntime as ort

    with open(meta_path) as fh:
        meta = json.load(fh)
    max_len = meta["max_len"]
    vocab = meta["vocab"]
    pad_idx = vocab.get("pad", 0)
    x_idx = vocab.get("X", pad_idx)
    none_idx = meta["none_idx"]
    labels = meta["labels"]

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
    probs_all = np.zeros((len(rows), len(labels)), dtype=np.float32) \
        if return_probs else None
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        x = np.asarray([encode(r["seq"]) for r in batch], dtype=np.int64)
        logits = sess.run(None, {inp_name: x})[0]
        logits -= logits.max(axis=-1, keepdims=True)
        e = np.exp(logits)
        probs = e / e.sum(axis=-1, keepdims=True)
        scores[start:start + len(batch)] = 1.0 - probs[:, none_idx]
        if return_probs:
            probs_all[start:start + len(batch)] = probs
    if return_probs:
        return scores, probs_all, labels
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
    """Dense per-sequence count vector. Used only for the small target
    matrix; per-row training matrices use _kmer_matrix_sparse instead
    (avoids materialising a dense N x 8000 matrix at >300k rows)."""
    v = np.zeros(len(index), dtype=np.float32)
    s = "".join(ch for ch in seq if ch in AA_SET)
    for i in range(len(s) - k + 1):
        j = index.get(s[i:i + k])
        if j is not None:
            v[j] += 1
    return v


def _kmer_matrix_sparse(rows, k, index):
    """Build a CSR sparse matrix of k-mer counts: shape (N, |index|).

    Density per row is ~1-3% (a sequence of length L has at most L-k+1
    distinct k-mers out of |index|=20**k), so for k=3 and ~300k rows
    this is ~50x smaller than the equivalent dense matrix. Returns a
    scipy.sparse.csr_matrix; sklearn's `LogisticRegression(solver=
    'liblinear')` and `scipy.sparse.linalg.norm` handle it natively."""
    from scipy import sparse
    indptr = [0]
    indices = []
    data = []
    for r in rows:
        counts = {}
        s = "".join(ch for ch in r["seq"] if ch in AA_SET)
        for i in range(len(s) - k + 1):
            j = index.get(s[i:i + k])
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
        for j, c in counts.items():
            indices.append(j)
            data.append(c)
        indptr.append(len(indices))
    return sparse.csr_matrix(
        (np.asarray(data, dtype=np.float32),
         np.asarray(indices, dtype=np.int32),
         np.asarray(indptr, dtype=np.int64)),
        shape=(len(rows), len(index)),
    )


def _l2_sparse(X):
    """Row-wise L2 normalize a CSR sparse matrix in-place-ish (returns
    a new CSR with scaled data array). Empty rows pass through as 0."""
    from scipy import sparse
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    inv = sparse.diags(1.0 / norms)
    return inv @ X


def _l2(X):
    """Dense L2 row-normalise; kept for the (small) target matrix."""
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1
    return X / n


def kmer_nn_scores(rows, target_seqs, k=3):
    """Max cosine similarity to any of the k WT target k-mer vectors.
    Targets are dense (|targets| is small); rows go through the sparse
    path because there can be hundreds of thousands of them."""
    index = build_kmer_index(k)
    T = np.stack([_kmer_vec(s, k, index) for s in target_seqs.values()])
    T = _l2(T)
    X = _l2_sparse(_kmer_matrix_sparse(rows, k, index))
    # sparse @ dense -> dense ndarray, no .todense() needed
    return (X @ T.T).max(axis=1)


def load_gate_mask(active_sites_dir, residue_maps_dir, variants_dir=None):
    """Build per-target gate lookup. For each target returns a dict:
      {"triples": list[(wt_idx, mpnn_0idx, expected_aa)],
       "wt_len":  int,    # length of the WT FASTA for this target
       "mpnn_len": int}   # length of MPNN designs (== PDB author range
                          #   hi-lo+1, matches parse_multiple_chains seq)

    The gate uses the query's length to decide which coordinate system
    to check against, avoiding coincidental single-letter matches at
    the "wrong" index."""
    import json
    from src.data.residue_map import ResidueMap
    out = {}
    for asp in sorted(Path(active_sites_dir).glob("*.json")):
        t = asp.stem
        rmp = Path(residue_maps_dir) / f"{t}.json"
        if not rmp.exists():
            continue
        active = json.loads(asp.read_text())
        fup = active.get("fixed_positions_uniprot", [])
        if not fup:
            continue
        rmap = ResidueMap.load(rmp)
        out[t] = {
            "triples": rmap.expected_for_gate(fup),
            "wt_len": rmap.wt_length,
            "mpnn_len": rmap.data.get("mpnn_length", 0),
        }
    return out


def apply_fixed_position_gate(rows, gate_mask, base_scores,
                               predicted_targets, target_conf_min=0.2):
    """Return a copy of base_scores where entries are zeroed whenever
    the query fails to preserve the predicted target's active-site
    residues in EITHER the WT or the MPNN coordinate system as a whole.

    Gate semantics: a query is "preserved" iff (all fixed positions
    match at wt_idx) OR (all fixed positions match at mpnn_0idx).
    Otherwise the gate fires. Per-position OR-ing lets coincidental
    single-letter matches at the "wrong" index mask real mutations
    elsewhere, so we check consistency at the coord-system level.

    predicted_targets[i] is a (target_id, confidence) tuple. Rows with
    confidence below target_conf_min are passed through unchanged."""
    scores = np.asarray(base_scores, dtype=np.float32).copy()
    for i, r in enumerate(rows):
        seq = r["seq"]
        t, conf = predicted_targets[i]
        if t is None or conf < target_conf_min:
            continue
        entry = gate_mask.get(t)
        if not entry or not entry["triples"]:
            continue
        # Pick coordinate system by length. Queries whose length doesn't
        # match either the WT or MPNN reference (decoys, scrambles,
        # perturb30 after length changes) are passed through - the gate
        # can only reliably check queries whose coordinate frame we
        # know.
        if len(seq) == entry["wt_len"]:
            idx_sel = lambda tr: tr[0]
        elif len(seq) == entry["mpnn_len"]:
            idx_sel = lambda tr: tr[1]
        else:
            continue
        for tr in entry["triples"]:
            j = idx_sel(tr)
            exp = tr[2]
            if j is None or not (0 <= j < len(seq)) or seq[j] != exp:
                scores[i] = 0.0
                break
    return scores


def capiti_predicted_targets(probs, labels):
    """For each row, return (top_non_none_target, prob_of_that_target).
    Uses the model's argmax over T1..T9 (ignoring 'none')."""
    none_idx = labels.index("none") if "none" in labels else -1
    out = []
    for row in probs:
        best_i, best_p = -1, -1.0
        for i, p in enumerate(row):
            if i == none_idx:
                continue
            if p > best_p:
                best_p = float(p); best_i = i
        t = labels[best_i] if best_i >= 0 else None
        out.append((t, best_p))
    return out


def kmer_lr_scores(train_rows, test_rows, k=3):
    """k-mer logistic-regression baseline. Uses sparse CSR matrices
    plus sklearn's `liblinear` solver (sparse-native, single-threaded
    but fast enough). Memory scales with number of nonzeros, not with
    N x |index|, so this comfortably handles 300k+ training rows."""
    index = build_kmer_index(k)
    Xtr = _l2_sparse(_kmer_matrix_sparse(train_rows, k, index))
    ytr = np.array([int(r["binary_label"]) for r in train_rows])
    Xte = _l2_sparse(_kmer_matrix_sparse(test_rows, k, index))
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear")
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]
