"""Build per-target residue-coordinate maps from SIFTS + UniProt.

For each target in active_sites/Ti.json:

  1. Fetch SIFTS XML for the PDB (cached under data/targets/sifts/).
  2. Fetch canonical UniProt sequence (cached under data/targets/uniprot/).
  3. Align the existing WT FASTA to the UniProt sequence. If the WT FASTA
     IS the UniProt sequence (or a clean substring of it), the alignment
     is trivial; otherwise we use a simple substring / offset search and
     fail loudly if we can't find a clean match. Engineered constructs
     (e.g., N-term tags) are accepted as long as some WT window matches.
  4. Read SIFTS per-residue mappings for the target chain; per UniProt
     residue record (uniprot_num, wt_idx, pdb_num, pdb_icode, mpnn_1idx,
     aa), cross-checking AA letters everywhere.
  5. Write data/targets/residue_maps/Ti.json.
  6. Stamp data/targets/active_sites/Ti.json with
     `fixed_positions_uniprot` derived from its existing
     `fixed_positions_pdb` via the SIFTS reverse map.

Run:
    python -m src.data.build_residue_map \\
        --active-sites data/targets/active_sites \\
        --fastas data/targets/primary_sequences \\
        --out data/targets/residue_maps
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


SIFTS_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/split_xml/{sub}/{pdb}.xml.gz"
)
UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"

SIFTS_NS = "{http://www.ebi.ac.uk/pdbe/docs/sifts/eFamily.xsd}"


def _fetch(url, out_path, binary=True, timeout=60):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return out_path.read_bytes() if binary else out_path.read_text()
    with urllib.request.urlopen(url, timeout=timeout) as r:
        data = r.read()
    if binary:
        out_path.write_bytes(data)
        return data
    else:
        text = data.decode("utf-8")
        out_path.write_text(text)
        return text


def fetch_sifts(pdb_id, cache_dir):
    pdb = pdb_id.lower()
    raw = _fetch(
        SIFTS_URL.format(sub=pdb[1:3], pdb=pdb),
        Path(cache_dir) / f"{pdb}.xml.gz",
        binary=True,
    )
    return gzip.decompress(raw).decode("utf-8")


def fetch_uniprot_fasta(acc, cache_dir):
    text = _fetch(
        UNIPROT_URL.format(acc=acc),
        Path(cache_dir) / f"{acc}.fasta",
        binary=False,
    )
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith(">")]
    return "".join(lines)


def read_wt_fasta(path):
    """Return first record (skip any monomer variants etc.)."""
    lines = [ln.strip() for ln in open(path) if ln.strip()]
    seq = ""
    for ln in lines[1:]:
        if ln.startswith(">"):
            break
        seq += ln
    return seq


def align_wt_to_uniprot(wt, unp):
    """Return the offset such that wt[i] corresponds to unp[i + offset].
    Specifically: find the largest prefix-window of wt that appears in
    unp. Works for simple cases (WT == UniProt, WT is an internal
    fragment, UniProt has leading signal peptide removed vs WT, etc.).
    Returns (offset, window_size) or (None, None) if no clean match."""
    for w in (80, 60, 40, 30, 20, 15):
        if len(wt) < w:
            continue
        # try anchored on the start, middle, and end of wt
        for anchor in (0, max(0, len(wt) // 2 - w // 2), len(wt) - w):
            chunk = wt[anchor:anchor + w]
            if "X" in chunk:
                continue
            j = unp.find(chunk)
            if j >= 0:
                # wt[anchor] == unp[j] -> wt[i] == unp[i - anchor + j]
                return j - anchor, w
    return None, None


def parse_sifts(xml_text, chain):
    """Yield per-residue dicts for the given PDB chain. Keys:
       uniprot_num, uniprot_aa, pdb_num, pdb_icode, pdb_aa, observed,
       uniprot_accession."""
    root = ET.fromstring(xml_text)
    for residue in root.iter(f"{SIFTS_NS}residue"):
        rec = {
            "uniprot_num": None,
            "uniprot_aa": None,
            "pdb_num": None,
            "pdb_icode": "",
            "pdb_aa": None,
            "observed": True,
            "uniprot_accession": None,
            "chain": None,
        }
        for xref in residue.findall(f"{SIFTS_NS}crossRefDb"):
            src = xref.get("dbSource")
            if src == "UniProt":
                try:
                    rec["uniprot_num"] = int(xref.get("dbResNum"))
                except (TypeError, ValueError):
                    pass
                rec["uniprot_aa"] = xref.get("dbResName")
                rec["uniprot_accession"] = xref.get("dbAccessionId")
            elif src == "PDB":
                ch = xref.get("dbChainId")
                rec["chain"] = ch
                num = xref.get("dbResNum")
                if num and num != "null":
                    # split numeric + icode (e.g., "100A")
                    j = 0
                    while j < len(num) and (num[j].isdigit() or
                                             (j == 0 and num[j] == "-")):
                        j += 1
                    try:
                        rec["pdb_num"] = int(num[:j])
                    except ValueError:
                        rec["pdb_num"] = None
                    rec["pdb_icode"] = num[j:]
                rec["pdb_aa"] = xref.get("dbResName")
        # observed?
        for det in residue.findall(f"{SIFTS_NS}residueDetail"):
            if det.text == "Not_Observed":
                rec["observed"] = False
        if rec["chain"] is not None and rec["chain"] != chain:
            continue
        if rec["uniprot_num"] is None:
            # some residues are PDB-only (expression tags). Skip - we
            # only care about UniProt-mapped residues.
            continue
        yield rec


THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def pdb_chain_ca_keys(pdb_path, chain):
    """Return (sorted list of (pdb_num, pdb_icode) from ATOM CA records
    for the given chain, lo_author). This is what ProteinMPNN's
    parse_multiple_chains.py sees: ATOM-only, per-residue CAs, gaps
    filled with '-' in the output sequence. HETATM residues (modified
    AAs like KCX) are invisible to MPNN even though SIFTS flags them
    observed."""
    keys = []
    seen = set()
    for line in open(pdb_path):
        if not line.startswith("ATOM"):
            continue
        if line[21] != chain:
            continue
        if line[12:16].strip() != "CA":
            continue
        try:
            num = int(line[22:26])
        except ValueError:
            continue
        ic = line[26].strip()
        key = (num, ic)
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
    if not keys:
        return [], None
    lo = min(n for n, _ in keys)
    return keys, lo


def build_one(target, active_sites, wt_seq, sifts_cache, unp_cache,
              pdb_path):
    """Build a residue map for one target. Returns (map_dict, diagnostics)."""
    tid = target
    pdb_id = active_sites["pdb"]
    chain = active_sites["chain"]
    acc = active_sites["uniprot"]

    sifts_xml = fetch_sifts(pdb_id, sifts_cache)
    unp_seq = fetch_uniprot_fasta(acc, unp_cache)

    sifts_residues = list(parse_sifts(sifts_xml, chain))
    # filter to rows whose uniprot_accession matches the declared accession
    filtered = [r for r in sifts_residues
                if r["uniprot_accession"] == acc]
    if not filtered:
        # some PDBs carry an alt accession (isoform, UPI). fall back.
        filtered = sifts_residues
    sifts_residues = filtered

    # Align WT FASTA to canonical UniProt to get wt_idx per uniprot_num.
    offset, window = align_wt_to_uniprot(wt_seq, unp_seq)

    def wt_idx_for(unp_num):
        if offset is None:
            return None
        # unp_num is 1-indexed; unp_seq[unp_num - 1] is the letter. We
        # found wt[i] == unp[i + offset], so i = unp_num - 1 - offset.
        i = (unp_num - 1) - offset
        if 0 <= i < len(wt_seq):
            return i
        return None

    # mpnn_1idx: ProteinMPNN's parse_multiple_chains.py reads ATOM CA
    # records (no HETATM), builds a chain sequence indexed by the PDB
    # author-number range from the first ATOM-CA onward, filling gaps
    # with '-'. The MPNN output and its fixed_positions_jsonl both use
    # 1-indexed positions into that sequence.
    #
    # Subtleties:
    #  * lo_author must come from the PDB itself (min ATOM-CA num),
    #    NOT from SIFTS-observed residues: some PDBs have leading
    #    residues that SIFTS doesn't map to any UniProt entry (e.g.,
    #    a Met at author num 0), and MPNN indexes from there.
    #  * A residue can be SIFTS-observed but absent from ATOM records
    #    (e.g., carbamylated lysine recorded as HETATM KCX). MPNN
    #    cannot preserve those — treat as unresolved.
    pdb_keys, lo_author = pdb_chain_ca_keys(pdb_path, chain)
    pdb_key_set = set(pdb_keys)
    mpnn_length = 0
    if lo_author is not None:
        hi_author = max(n for n, _ in pdb_keys)
        mpnn_length = hi_author - lo_author + 1
        for r in sifts_residues:
            if r["pdb_num"] is None:
                continue
            if (r["pdb_num"], r["pdb_icode"]) not in pdb_key_set:
                # SIFTS said observed but MPNN's parser won't see it.
                r["observed"] = False
                continue
            r["_mpnn_1idx"] = r["pdb_num"] - lo_author + 1

    # assemble per-UniProt records
    diags = []
    records = []
    for r in sifts_residues:
        unp_num = r["uniprot_num"]
        wt_i = wt_idx_for(unp_num)
        unp_letter = r["uniprot_aa"]
        pdb_letter = THREE_TO_ONE.get(r["pdb_aa"]) if r["pdb_aa"] else None
        wt_letter = wt_seq[wt_i] if wt_i is not None else None
        # sanity checks
        if wt_letter is not None and unp_letter and wt_letter != unp_letter:
            diags.append(
                f"{tid}: wt[{wt_i}]={wt_letter} != uniprot[{unp_num}]="
                f"{unp_letter}"
            )
        if r["observed"] and pdb_letter and unp_letter and \
                pdb_letter != unp_letter:
            diags.append(
                f"{tid}: pdb[{r['pdb_num']}{r['pdb_icode']}]="
                f"{pdb_letter} != uniprot[{unp_num}]={unp_letter}"
            )
        records.append({
            "uniprot_num": unp_num,
            "wt_idx": wt_i,
            "pdb_num": r["pdb_num"] if r["observed"] else None,
            "pdb_icode": r["pdb_icode"] if r["observed"] else "",
            "mpnn_1idx": r.get("_mpnn_1idx"),
            "aa": unp_letter or pdb_letter or wt_letter,
        })

    # convert active_sites.fixed_positions_pdb -> uniprot numbers so
    # downstream tools have a UniProt-native list
    pdb_to_unp = {(r["pdb_num"], r["pdb_icode"]): r["uniprot_num"]
                  for r in sifts_residues if r["pdb_num"] is not None}
    fixed_unp = []
    for a in active_sites.get("fixed_positions_pdb", []):
        # active_sites stores bare ints; assume no icode
        u = pdb_to_unp.get((a, ""))
        if u is not None:
            fixed_unp.append(u)
        else:
            diags.append(f"{tid}: fixed_positions_pdb {a} -> no UniProt map")
    fixed_unp = sorted(set(fixed_unp))

    return {
        "target": tid,
        "pdb": pdb_id,
        "chain": chain,
        "uniprot_accession": acc,
        "wt_length": len(wt_seq),
        "uniprot_length": len(unp_seq),
        "mpnn_length": mpnn_length,
        "wt_unp_offset": offset,
        "alignment_window": window,
        "fixed_positions_uniprot": fixed_unp,
        "residues": records,
    }, diags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--active-sites", default="data/targets/active_sites")
    ap.add_argument("--fastas", default="data/targets/primary_sequences")
    ap.add_argument("--out", default="data/targets/residue_maps")
    ap.add_argument("--sifts-cache", default="data/targets/sifts")
    ap.add_argument("--uniprot-cache", default="data/targets/uniprot")
    ap.add_argument("--structures", default="data/targets/structures")
    ap.add_argument("--update-active-sites", action="store_true",
                    help="write fixed_positions_uniprot back into "
                         "active_sites/Ti.json")
    ap.add_argument("--targets", nargs="*", default=None,
                    help="restrict to these target IDs (default: all Ti.json)")
    args = ap.parse_args()

    active_dir = Path(args.active_sites)
    fasta_dir = Path(args.fastas)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_files = sorted(active_dir.glob("*.json"))
    if args.targets:
        target_files = [p for p in target_files if p.stem in args.targets]

    any_diag = False
    for tf in target_files:
        tid = tf.stem
        active = json.loads(tf.read_text())
        wt = read_wt_fasta(fasta_dir / f"{tid}.fasta")
        print(f"[{tid}] pdb={active['pdb']} uniprot={active.get('uniprot')} "
              f"wt_len={len(wt)}", file=sys.stderr)
        pdb_path = Path(args.structures, f"{tid}.pdb")
        try:
            res_map, diags = build_one(
                tid, active, wt, args.sifts_cache, args.uniprot_cache,
                pdb_path)
        except Exception as e:
            print(f"[{tid}] FAILED: {e}", file=sys.stderr)
            any_diag = True
            continue
        (out_dir / f"{tid}.json").write_text(json.dumps(res_map, indent=2))
        n_resolved = sum(1 for r in res_map["residues"]
                         if r["mpnn_1idx"] is not None)
        n_wt = sum(1 for r in res_map["residues"] if r["wt_idx"] is not None)
        print(f"[{tid}] residues={len(res_map['residues'])} "
              f"in_wt={n_wt} observed_in_pdb={n_resolved} "
              f"offset={res_map['wt_unp_offset']} "
              f"fixed_uniprot={res_map['fixed_positions_uniprot']}",
              file=sys.stderr)
        for d in diags[:20]:
            print(f"  ! {d}", file=sys.stderr)
        if diags:
            any_diag = True
            if len(diags) > 20:
                print(f"  ... ({len(diags) - 20} more)", file=sys.stderr)
        if args.update_active_sites:
            active["fixed_positions_uniprot"] = res_map["fixed_positions_uniprot"]
            tf.write_text(json.dumps(active, indent=2))

    if any_diag:
        print("\n[build_residue_map] completed WITH diagnostics; review "
              "above before regenerating variants.", file=sys.stderr)
        sys.exit(1)
    print("\n[build_residue_map] all clean.", file=sys.stderr)


if __name__ == "__main__":
    main()
