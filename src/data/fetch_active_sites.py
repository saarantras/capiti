"""Fetch active/binding/metal/disulfide residues for a set of PDB IDs.

For each PDB ID:
  1. SIFTS lookup -> UniProt accession(s) + residue mapping (UniProt <-> PDB).
  2. UniProt features -> positions of ACT_SITE, BINDING, METAL, DISULFID, SITE.
  3. Map UniProt residue numbers back to PDB author residue numbers (chain A)
     using the SIFTS mapping.

Writes one JSON per target to data/targets/active_sites/<TID>.json with:
  {
    "target": "T2", "pdb": "2OV5", "uniprot": "P62593",
    "chain": "A",
    "fixed_positions_pdb": [70, 73, 130, 166, ...],   # PDB author numbering
    "features": [{"type": "ACT_SITE", "uniprot_pos": 70, "pdb_pos": 70,
                  "description": "Acyl-ester intermediate"}, ...],
    "notes": "..."
  }

No ML deps needed. Uses SIFTS (EBI) + UniProt REST.
"""
import argparse, json, pathlib, sys, urllib.request, urllib.error


FEATURE_TYPES = {"active site", "binding site", "metal binding", "disulfide bond", "site"}


def fetch_json(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


def residue_listing(pdb_id, chain):
    """Per-residue SEQRES residue_number <-> author_residue_number for a chain."""
    url = f"https://www.ebi.ac.uk/pdbe/api/pdb/entry/residue_listing/{pdb_id.lower()}/chain/{chain}"
    try:
        d = fetch_json(url)
    except Exception:
        return []
    out = []
    for mol in d.get(pdb_id.lower(), {}).get("molecules", []):
        for ch in mol.get("chains", []):
            if ch.get("chain_id") != chain:
                continue
            for r in ch.get("residues", []):
                out.append((r.get("residue_number"), r.get("author_residue_number")))
    return out


def sifts_mapping(pdb_id):
    """Returns list of dicts, one per (uniprot, chain) segment:
       {unp_acc, chain, unp_start, unp_end, pdb_start, pdb_end}
       PDB numbering returned is the *author* residue number.
    """
    pdb_id = pdb_id.lower()
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}"
    data = fetch_json(url)
    segs = []
    for acc, info in data.get(pdb_id, {}).get("UniProt", {}).items():
        for m in info["mappings"]:
            pstart = m["start"]["author_residue_number"]
            pend = m["end"]["author_residue_number"]
            unp_start = m["unp_start"]
            unp_end = m["unp_end"]
            # prefer a non-null anchor to compute offset; they should agree
            offset = None
            if pstart is not None and unp_start is not None:
                offset = pstart - unp_start
            elif pend is not None and unp_end is not None:
                offset = pend - unp_end
            segs.append(dict(
                unp_acc=acc,
                chain=m["chain_id"],
                unp_start=unp_start,
                unp_end=unp_end,
                pdb_start=pstart,
                pdb_end=pend,
                offset=offset,
            ))
    return segs


def uniprot_features(acc):
    url = f"https://rest.uniprot.org/uniprotkb/{acc}.json"
    data = fetch_json(url)
    out = []
    for feat in data.get("features", []):
        if feat.get("type", "").lower() in FEATURE_TYPES:
            loc = feat.get("location", {})
            start = loc.get("start", {}).get("value")
            end = loc.get("end", {}).get("value")
            desc = feat.get("description", "")
            out.append(dict(type=feat["type"], start=start, end=end, description=desc))
    return out


def unp_to_pdb(unp_pos, segments, chain="A"):
    """Map a UniProt residue number to PDB author numbering on `chain`.
       Returns None if outside any mapped segment on that chain.
    """
    for s in segments:
        if s["chain"] != chain:
            continue
        if s["offset"] is None or s["unp_start"] is None:
            continue
        if s["unp_start"] <= unp_pos <= s["unp_end"]:
            return unp_pos + s["offset"]
    return None


def process(target_id, pdb_id, chain, out_dir):
    rec = dict(target=target_id, pdb=pdb_id, chain=chain,
               uniprot=None, features=[], fixed_positions_pdb=[], notes="")
    try:
        segs = sifts_mapping(pdb_id)
    except Exception as e:
        rec["notes"] = f"SIFTS lookup failed: {e}"
        return rec
    chain_segs = [s for s in segs if s["chain"] == chain]
    if not chain_segs:
        # try any chain if requested chain missing
        if segs:
            chain = segs[0]["chain"]
            chain_segs = [s for s in segs if s["chain"] == chain]
            rec["chain"] = chain
            rec["notes"] += f"chain A not in SIFTS; using {chain}. "
        else:
            rec["notes"] += "no SIFTS segments. "
            return rec
    # fallback: if any segment lacks an offset, recover from residue_listing.
    # SIFTS convention: SEQRES residue_number=1 corresponds to the segment's unp_start.
    # So offset = author(residue_number=1) - unp_start.
    if any(s["offset"] is None for s in chain_segs):
        listing = residue_listing(pdb_id, chain)
        res1 = next((a for rn, a in listing if rn == 1 and a is not None), None) if listing else None
        if res1 is not None:
            for s in chain_segs:
                if s["offset"] is None and s.get("unp_start") is not None:
                    s["offset"] = res1 - s["unp_start"]
    # take the UniProt acc with the longest total coverage on this chain
    coverage = {}
    for s in chain_segs:
        coverage[s["unp_acc"]] = coverage.get(s["unp_acc"], 0) + (s["unp_end"] - s["unp_start"] + 1)
    acc = max(coverage, key=coverage.get)
    rec["uniprot"] = acc
    try:
        feats = uniprot_features(acc)
    except Exception as e:
        rec["notes"] += f"UniProt fetch failed: {e}"
        return rec
    fixed_pdb = set()
    fixed_unp = set()
    for f in feats:
        s, e = f.get("start"), f.get("end")
        if s is None or e is None:
            continue
        ftype = (f.get("type") or "").lower()
        # Disulfide bonds are encoded as a single feature spanning the
        # two bonded cysteines (start..end), but only the endpoints are
        # the residues that must be preserved - the residues between
        # are unrelated to the bond.
        if ftype == "disulfide bond" and s != e:
            positions = (s, e)
        else:
            positions = range(s, e + 1)
        for p in positions:
            fixed_unp.add(p)
            mapped = unp_to_pdb(p, segs, chain=chain)
            if mapped is not None:
                fixed_pdb.add(mapped)
        rec["features"].append(dict(type=f["type"], unp_start=s, unp_end=e,
                                    description=f["description"]))
    rec["fixed_positions_uniprot"] = sorted(fixed_unp)
    rec["fixed_positions_pdb"] = sorted(fixed_pdb)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True,
                    help="TSV with columns: target_id pdb_id [chain]")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for ln in open(args.targets):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        tid, pdb = parts[0], parts[1]
        chain = parts[2] if len(parts) > 2 else "A"
        rows.append((tid, pdb, chain))

    for tid, pdb, chain in rows:
        print(f"[{tid}] {pdb} chain {chain} ...", flush=True)
        rec = process(tid, pdb, chain, out)
        (out / f"{tid}.json").write_text(json.dumps(rec, indent=2))
        n = len(rec["fixed_positions_pdb"])
        unp = rec["uniprot"] or "-"
        note = (" [" + rec["notes"].strip() + "]") if rec["notes"] else ""
        print(f"  uniprot={unp}  n_fixed={n}  n_features={len(rec['features'])}{note}",
              flush=True)


if __name__ == "__main__":
    sys.exit(main())
