"""capiti CLI: nucleotide sequence -> TRUE/FALSE vs in-set function.

Usage:
    capiti <nt_sequence> [--cutoff 0.5] [-v] [--which]
    capiti --fasta seqs.fa [--cutoff 0.5]
    capiti --stdin [--cutoff 0.5]

Model: bundled with the package under capiti/_model/. Override with
--model / --meta, or set CAPITI_MODEL (and optionally CAPITI_META).

Exit code: 0 if any input is TRUE at the cutoff, 1 otherwise. Handy for
shell pipelines.
"""
from __future__ import annotations
import argparse, json, os, sys
from importlib import resources


# Standard genetic code (RNA/DNA both accepted; U normalized to T).
CODONS = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V',
    'TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
    'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
    'AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E',
    'TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
    'AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G',
}


def translate(nt: str) -> str:
    nt = "".join(nt.upper().split()).replace("U", "T")
    aa = []
    for i in range(0, len(nt) - 2, 3):
        c = nt[i:i+3]
        if len(c) < 3:
            break
        r = CODONS.get(c, "X")
        if r == "*":
            break
        aa.append(r)
    return "".join(aa)


def encode(seq, aa_to_idx, pad_idx, max_len):
    x = [pad_idx] * max_len
    x_idx = aa_to_idx.get("X", pad_idx)
    for i, c in enumerate(seq[:max_len]):
        x[i] = aa_to_idx.get(c, x_idx)
    return x


def softmax(x):
    import numpy as np
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def read_fasta(path):
    name, buf = None, []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                yield name, "".join(buf)
            name = line[1:].split()[0]
            buf = []
        else:
            buf.append(line)
    if name is not None:
        yield name, "".join(buf)


def _bundled(filename):
    """Return a path to a file bundled under capiti/_model/."""
    return resources.files("capiti").joinpath("_model", filename)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="capiti",
        description="Predict whether a nucleotide sequence encodes an "
                    "in-set enzymatic function.",
    )
    ap.add_argument("sequence", nargs="?", help="nucleotide sequence (ACGT[U])")
    ap.add_argument("--fasta", help="read nucleotide sequences from FASTA")
    ap.add_argument("--stdin", action="store_true",
                    help="read one sequence from stdin")
    ap.add_argument("--cutoff", type=float, default=0.5,
                    help="probability threshold for TRUE (default 0.5)")
    ap.add_argument("--model", default=os.environ.get("CAPITI_MODEL"),
                    help="path to .onnx model (default: bundled, env: CAPITI_MODEL)")
    ap.add_argument("--meta", default=os.environ.get("CAPITI_META"),
                    help="path to .meta.json (default: bundled, env: CAPITI_META)")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--which", action="store_true",
                    help="also report top-class label (T1..T9 or none)")
    ap.add_argument("-V", "--version", action="store_true")
    args = ap.parse_args(argv)

    if args.version:
        from capiti import __version__
        print(f"capiti {__version__}")
        return 0

    # lazy imports
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as e:
        sys.stderr.write(f"missing dependency: {e}\n"
                         "install: pip install 'capiti[all]' or "
                         "pip install onnxruntime numpy\n")
        return 2

    # silence onnxruntime's environment-level warnings (GPU discovery etc.)
    # on CPU-only devices. override with CAPITI_LOG_LEVEL=0..4.
    log_lvl = int(os.environ.get("CAPITI_LOG_LEVEL", "3"))
    try:
        ort.set_default_logger_severity(log_lvl)
    except Exception:
        pass

    model_path = args.model or str(_bundled("capiti.onnx"))
    meta_path = args.meta or str(_bundled("capiti.meta.json"))

    with open(meta_path) as fh:
        meta = json.load(fh)
    max_len = meta["max_len"]
    aa_to_idx = meta["vocab"]
    pad_idx = aa_to_idx.get("pad", 0)
    labels = meta["labels"]
    none_idx = meta["none_idx"]

    # respect tight thread budgets on small devices
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.environ.get("CAPITI_THREADS", "1"))
    so.inter_op_num_threads = 1
    # suppress onnxruntime's noisy GPU-discovery warnings on CPU-only devices
    # (e.g. Raspberry Pi) where /sys/class/drm is absent.
    so.log_severity_level = int(os.environ.get("CAPITI_LOG_LEVEL", "3"))
    sess = ort.InferenceSession(model_path, sess_options=so,
                                 providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name

    # gather inputs
    items = []
    if args.fasta:
        items = list(read_fasta(args.fasta))
    elif args.stdin:
        items = [("stdin", sys.stdin.read().strip())]
    elif args.sequence:
        items = [("seq", args.sequence)]
    else:
        ap.error("provide a sequence, --fasta, or --stdin")

    toks = []
    metas = []
    for name, nt in items:
        aa = translate(nt)
        ids = encode(aa, aa_to_idx, pad_idx, max_len)
        toks.append(ids)
        metas.append((name, nt, aa))
    x = np.asarray(toks, dtype=np.int64)

    logits = sess.run(None, {inp_name: x})[0]
    probs = softmax(logits)
    inset = 1.0 - probs[:, none_idx]
    top = probs.argmax(axis=-1)

    any_true = False
    for (name, nt, aa), p_inset, top_idx in zip(metas, inset, top):
        is_true = bool(p_inset >= args.cutoff)
        any_true = any_true or is_true
        flag = "TRUE" if is_true else "FALSE"
        if args.verbose or args.fasta:
            extra = f"  p_inset={p_inset:.3f}"
            if args.which:
                extra += f"  top={labels[int(top_idx)]}"
            if args.fasta:
                print(f">{name}\t{flag}{extra}")
            else:
                print(f"{flag}{extra}")
        else:
            print(flag)

    return 0 if any_true else 1


if __name__ == "__main__":
    sys.exit(main())
