"""Canonical sequence utilities: codon table, translation, ORF finding.

Single source of truth for sequence operations across the package and the
`capiti-orf` / `capiti-translate` console scripts. `capiti.cli` imports
`translate` from here rather than carrying its own copy.

Standard genetic code; RNA `U` is normalized to DNA `T` on input.
"""
from __future__ import annotations
import argparse
import sys
from typing import Iterable, Iterator, List, Tuple


# Standard genetic code. `*` denotes a stop codon.
CODON_TABLE: dict[str, str] = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V',
    'TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
    'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
    'AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E',
    'TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
    'AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G',
}
START_CODON = "ATG"
STOP_CODONS = frozenset({"TAA", "TAG", "TGA"})


def normalize_nt(nt: str) -> str:
    """Uppercase, strip whitespace, normalize U -> T."""
    return "".join(nt.upper().split()).replace("U", "T")


def translate(nt: str, *, unknown: str = "X", stop_at_first: bool = True) -> str:
    """Translate nucleotides to amino acids in the +1 frame.

    Stops at first stop codon when stop_at_first=True (the default; matches
    the on-device CLI). Incomplete trailing codons are skipped. Codons
    containing ambiguity (e.g. N) become `unknown`.
    """
    nt = normalize_nt(nt)
    aa: list[str] = []
    for i in range(0, len(nt) - 2, 3):
        codon = nt[i:i+3]
        r = CODON_TABLE.get(codon, unknown)
        if r == "*":
            if stop_at_first:
                break
            aa.append("*")
            continue
        aa.append(r)
    return "".join(aa)


def find_orfs(nt: str, *, min_len: int = 0,
              both_strands: bool = False) -> List[Tuple[int, str, str]]:
    """Find ORFs (ATG -> first in-frame stop, inclusive) in the input.

    Searches all three frames on the forward strand; with both_strands=True,
    also searches the reverse complement (frames are reported relative to
    that strand). Each result is `(start, strand, sequence)` where strand is
    "+" or "-" and `start` is the 0-based ATG offset on that strand.

    `min_len` is the minimum nt length of the returned ORF (including the
    stop codon if present). Tiny ORFs from internal ATGs are filtered here.
    """
    nt = normalize_nt(nt)
    strands = [("+", nt)]
    if both_strands:
        strands.append(("-", _revcomp(nt)))
    out: list[tuple[int, str, str]] = []
    for strand, s in strands:
        out.extend((i, strand, orf) for i, orf in _scan_forward(s)
                   if len(orf) >= min_len)
    return out


def _scan_forward(seq: str) -> Iterator[Tuple[int, str]]:
    """Yield (start_offset, orf_sequence) for every ATG-to-stop ORF in any frame."""
    pos = 0
    while True:
        start = seq.find(START_CODON, pos)
        if start == -1:
            return
        stop_pos = None
        for i in range(start, len(seq) - 2, 3):
            if seq[i:i+3] in STOP_CODONS:
                stop_pos = i + 3
                break
        yield start, seq[start:stop_pos] if stop_pos is not None else seq[start:]
        pos = start + 1


_COMPLEMENT = str.maketrans("ACGTUNacgtun", "TGCAANtgcaan")


def _revcomp(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


# ---------- shared FASTA-style I/O for the CLIs ----------

def _iter_fasta(fh) -> Iterator[Tuple[str, str]]:
    name, buf = None, []
    for line in fh:
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


def _read_inputs(args) -> List[Tuple[str, str]]:
    if args.fasta:
        path = args.fasta
        if path == "-":
            return list(_iter_fasta(sys.stdin))
        with open(path) as fh:
            return list(_iter_fasta(fh))
    if args.stdin:
        data = sys.stdin.read()
        if data.lstrip().startswith(">"):
            from io import StringIO
            return list(_iter_fasta(StringIO(data)))
        return [("seq", data.strip())]
    if args.sequence:
        return [("seq", args.sequence)]
    return []


def _add_input_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("sequence", nargs="?",
                    help="nucleotide sequence (ACGT[U])")
    ap.add_argument("--fasta",
                    help="read FASTA from path (`-` for stdin)")
    ap.add_argument("--stdin", action="store_true",
                    help="read a single sequence (or FASTA) from stdin")


# ---------- CLI entry points ----------

def orf_main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="capiti-orf",
        description="Find ORFs (ATG -> first in-frame stop) in nucleotide "
                    "sequences. Emits FASTA on stdout.",
    )
    _add_input_args(ap)
    ap.add_argument("--min-len", type=int, default=0,
                    help="minimum ORF length in nt, including stop codon "
                         "(default 0 = no filter)")
    ap.add_argument("--both-strands", action="store_true",
                    help="also scan the reverse complement")
    args = ap.parse_args(argv)

    items = _read_inputs(args)
    if not items:
        ap.error("provide a sequence, --fasta, or --stdin")

    n = 0
    for name, nt in items:
        for i, strand, orf in find_orfs(nt, min_len=args.min_len,
                                         both_strands=args.both_strands):
            n += 1
            sys.stdout.write(f">{name}|orf{n}|{strand}|{i}|{len(orf)}\n{orf}\n")
    return 0 if n else 1


def translate_main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="capiti-translate",
        description="Translate nucleotides to amino acids in frame +1. "
                    "Stops at the first stop codon. Emits FASTA on stdout.",
    )
    _add_input_args(ap)
    ap.add_argument("--unknown", default="X",
                    help="symbol for codons not in the standard table "
                         "(default X)")
    ap.add_argument("--read-through", action="store_true",
                    help="continue past stop codons (emits `*`)")
    args = ap.parse_args(argv)

    items = _read_inputs(args)
    if not items:
        ap.error("provide a sequence, --fasta, or --stdin")

    for name, nt in items:
        aa = translate(nt, unknown=args.unknown,
                       stop_at_first=not args.read_through)
        sys.stdout.write(f">{name}\n{aa}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(orf_main())
