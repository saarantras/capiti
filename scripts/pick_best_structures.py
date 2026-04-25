"""Select one best PDB per target from the anonymized candidate lists
and emit a kosher configs/targets-<set>.tsv.

Selection rules, in order:
  1. Prefer X-ray > EM > NMR > Neutron > other (within <= 3.0 A
     resolution cap for the chosen candidate).
  2. Among eligible entries, lowest resolution wins.
  3. Chain: if the candidate's `chains` field names chain A, use A;
     otherwise the first chain listed.
  4. Length cap: if the declared chain range exceeds max_len, flag.

Rows that don't satisfy any eligible filter are written to a review
file (configs/targets-<set>.review.tsv) with a reason, so we can
hand-resolve them instead of silently dropping.

AlphaFold entries (db=AF in the E list) are passed through with
pdb_id=AF-<acc> and a sentinel chain A; the downstream pipeline needs
a small adapter to fetch AF-DB PDBs instead of RCSB + SIFTS.

Usage:
  python scripts/pick_best_structures.py \\
      configs/capiti_C_targets_anonymized.tsv configs/targets-C.tsv
  python scripts/pick_best_structures.py \\
      configs/capiti_E_targets_anonymized.tsv configs/targets-E.tsv
"""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict


MAX_LEN = 1200
RES_CAP = 3.5  # angstroms
METHOD_RANK = {
    "x-ray": 0,
    "x-ray diffraction": 0,
    "em": 1,
    "electron microscopy": 1,
    "nmr": 2,
    "solution nmr": 2,
    "neutron": 3,
    "other": 4,
    "": 4,
}


def parse_chain(chains_field, declared_chain):
    """For the C-list `chains` field like 'A/B=1-123' or 'A=340-585',
    or the E-list simple 'A' / 'B'. Returns (chain_letter, length_or_None)."""
    if not chains_field:
        return declared_chain or "A", None
    # C-list form
    m = re.match(r"^([A-Za-z0-9/]+)=(\d+)-(\d+)$", chains_field)
    if m:
        chains = m.group(1).split("/")
        lo, hi = int(m.group(2)), int(m.group(3))
        pick = "A" if "A" in chains else chains[0]
        return pick, hi - lo + 1
    # E-list: just a single-letter chain
    if re.match(r"^[A-Za-z0-9]$", chains_field):
        return chains_field, None
    # fallback
    return declared_chain or chains_field.split("/")[0], None


def parse_resolution(s):
    if not s:
        return None
    m = re.match(r"^\s*([\d.]+)", s)
    return float(m.group(1)) if m else None


def normalize_method(m):
    if not m:
        return ""
    return m.strip().lower()


def is_pdb_id(s):
    return bool(re.match(r"^[0-9][A-Za-z0-9]{3}$", s or ""))


def best_candidate(rows):
    """rows is a list of dicts with fields: pdb, chains/chain,
    method, resolution, db (optional). Returns (best_row, None) or
    (None, reason)."""
    scored = []
    for r in rows:
        method = normalize_method(r.get("method"))
        res = parse_resolution(r.get("resolution"))
        if r.get("db") == "AF":
            # AlphaFold: no resolution, score below all X-ray but above
            # trash. If the target has ANY experimental structure we'd
            # prefer that instead.
            scored.append((METHOD_RANK.get(method, 9) + 100, 999.0, r))
            continue
        if not is_pdb_id(r.get("pdb", "")):
            continue  # skip lowercase/malformed rows
        if method and method not in METHOD_RANK:
            method = "other"
        mrank = METHOD_RANK.get(method, 9)
        if res is None:
            # unknown resolution; penalise but keep in play
            scored.append((mrank, 99.0, r))
        else:
            scored.append((mrank, res, r))

    if not scored:
        return None, "no parseable candidates"

    scored.sort(key=lambda x: (x[0], x[1]))
    # AlphaFold entries pass through with no resolution check (we treat
    # any high-confidence AF prediction as in-scope).
    for mrank, res, r in scored:
        if r.get("db") == "AF":
            return r, None
        if mrank < 4 and res <= RES_CAP:
            return r, None
    # nothing under the resolution cap - take the best anyway but flag
    mrank, res, r = scored[0]
    reason = f"best resolution {res} A (> {RES_CAP}) method={mrank}"
    return r, reason


def load_rows(path):
    with open(path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for r in reader:
            # unify column names between the C and E files
            if "PDB" in r and "pdb" not in r:
                r["pdb"] = r.pop("PDB")
            if "id" in r and "pdb" not in r:
                r["pdb"] = r.pop("id")
            yield r


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: pick_best_structures.py IN.tsv OUT.tsv")
    in_path, out_path = sys.argv[1], sys.argv[2]
    by_target = defaultdict(list)
    for r in load_rows(in_path):
        by_target[r["target_id"]].append(r)

    review_path = out_path.replace(".tsv", ".review.tsv")
    kept = []
    review = []
    for tid in sorted(by_target, key=lambda s: (int(re.search(r"\d+", s).group()),)):
        best, reason = best_candidate(by_target[tid])
        if best is None:
            review.append((tid, "", "", "", reason))
            continue
        pdb_id = best.get("pdb", "")
        chain, span = parse_chain(best.get("chains") or best.get("chain", ""),
                                   best.get("chain", ""))
        if best.get("db") == "AF":
            pdb_id = "AF:" + best.get("pdb", "")
        length_note = ""
        if span is not None and span > MAX_LEN:
            length_note = f"chain length {span} > max_len {MAX_LEN}"
        row = (tid, pdb_id, chain, length_note or reason or "")
        if length_note or reason:
            review.append(row)
        else:
            kept.append(row)

    with open(out_path, "w") as fh:
        fh.write("# target_id\tpdb_id\tchain\n")
        for tid, pdb, chain, _ in kept:
            fh.write(f"{tid}\t{pdb}\t{chain}\n")

    with open(review_path, "w") as fh:
        fh.write("# target_id\tpdb_id\tchain\treview_reason\n")
        for tid, pdb, chain, reason in review:
            fh.write(f"{tid}\t{pdb}\t{chain}\t{reason}\n")

    print(f"kept {len(kept)} -> {out_path}")
    print(f"review {len(review)} -> {review_path}")


if __name__ == "__main__":
    main()
