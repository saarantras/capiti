"""One-shot orchestrator for bringing a new target set online.

Given `configs/targets-<SET>.tsv` (# target_id pdb_id chain), writes
per-set artifacts under `data/<SET>/targets/`:

  structures/<Ti>.pdb            downloaded from RCSB (pdb_id) or AFDB
                                 (if pdb_id == "AF:<uniprot>")
  primary_sequences/<Ti>.fasta   UniProt canonical sequence for the
                                 target's UniProt accession
  active_sites/<Ti>.json         via src.data.fetch_active_sites
  residue_maps/<Ti>.json         via src.data.build_residue_map

Skips work that's already done (idempotent).

Usage:
  python scripts/setup_set.py --set C --targets-tsv configs/targets-C.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


_UA = "capiti-setup/0.1 (https://github.com/mcnoonz/capiti)"


def _open_with_retry(url, timeout=30, tries=5, base_delay=1.0):
    """urlopen wrapper: sets a UA, sleeps 100ms between calls to dodge
    rate limits, retries on 429/503 with exponential backoff."""
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    delay = base_delay
    last = None
    for attempt in range(tries):
        try:
            time.sleep(0.1)  # polite pacing
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 504) and attempt < tries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise
        except urllib.error.URLError as e:
            last = e
            if attempt < tries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise last  # unreachable

RCSB_URL = "https://files.rcsb.org/download/{pdb}.pdb"
AF_API = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"
UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"


def af_pdb_url(acc):
    """Ask the AFDB API for the current model file URL - AFDB version
    changes over time (v4 -> v5 -> v6 ...) and hardcoding breaks."""
    with _open_with_retry(AF_API.format(acc=acc), timeout=30) as r:
        entries = json.loads(r.read())
    if not entries:
        raise RuntimeError(f"no AFDB prediction for {acc}")
    url = entries[0].get("pdbUrl")
    if not url:
        raise RuntimeError(f"no pdbUrl in AFDB entry for {acc}")
    return url


def fetch(url, out_path, timeout=60):
    out_path = Path(out_path)
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with _open_with_retry(url, timeout=timeout) as r:
        out_path.write_bytes(r.read())


def sifts_uniprot(pdb_id):
    """Given a 4-letter PDB ID, look up primary UniProt accession via
    the PDBe REST mapping endpoint. Returns accession or None."""
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id.lower()}"
    try:
        with _open_with_retry(url, timeout=30) as r:
            d = json.loads(r.read())
    except Exception:
        return None
    entry = d.get(pdb_id.lower(), {})
    unp = entry.get("UniProt") or {}
    if not unp:
        return None
    # take accession with the longest coverage
    best = None
    best_cov = -1
    for acc, info in unp.items():
        cov = sum(m["unp_end"] - m["unp_start"] + 1
                   for m in info.get("mappings", []))
        if cov > best_cov:
            best_cov = cov
            best = acc
    return best


def uniprot_fasta(acc, cache_path):
    fetch(UNIPROT_FASTA_URL.format(acc=acc), cache_path)
    lines = [ln for ln in cache_path.read_text().splitlines()
             if ln and not ln.startswith(">")]
    return "".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="set_name", required=True)
    ap.add_argument("--targets-tsv", required=True)
    ap.add_argument("--root", default=None,
                    help="base dir (default: data/<set>)")
    args = ap.parse_args()

    root = Path(args.root or f"data/{args.set_name}")
    targets_dir = root / "targets"
    struct_dir = targets_dir / "structures"
    fasta_dir = targets_dir / "primary_sequences"
    active_dir = targets_dir / "active_sites"
    residue_dir = targets_dir / "residue_maps"
    sifts_dir = targets_dir / "sifts"
    unp_dir = targets_dir / "uniprot"
    for d in (struct_dir, fasta_dir, active_dir, residue_dir, sifts_dir,
              unp_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. per-target structure + WT fasta
    rows = []
    for ln in open(args.targets_tsv):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        tid = parts[0]
        pdb_id = parts[1]
        chain = parts[2] if len(parts) > 2 else "A"
        rows.append((tid, pdb_id, chain))

    n_fail = 0
    for tid, pdb_id, chain in rows:
        struct_path = struct_dir / f"{tid}.pdb"
        fasta_path = fasta_dir / f"{tid}.fasta"
        try:
            if pdb_id.startswith("AF:"):
                acc = pdb_id.split(":", 1)[1]
                if not struct_path.exists():
                    fetch(af_pdb_url(acc), struct_path)
            else:
                if not struct_path.exists():
                    fetch(RCSB_URL.format(pdb=pdb_id.lower()), struct_path)
                acc = sifts_uniprot(pdb_id)
                if acc is None:
                    print(f"[{tid}] no SIFTS UniProt mapping; skipping",
                          file=sys.stderr)
                    n_fail += 1
                    continue
            if not fasta_path.exists():
                seq = uniprot_fasta(acc, unp_dir / f"{acc}.fasta")
                fasta_path.write_text(f">{tid}\n{seq}\n")
            print(f"[{tid}] {pdb_id} chain {chain} acc={acc}", flush=True)
        except Exception as e:
            print(f"[{tid}] FAIL {pdb_id}: {e}", file=sys.stderr)
            n_fail += 1

    print(f"\n[setup] structures + WT FASTAs done "
          f"({len(rows) - n_fail}/{len(rows)} ok)")

    # 2. active sites (fetch_active_sites handles AF: prefix)
    subprocess.run(
        ["python", "-m", "src.data.fetch_active_sites",
         "--targets", args.targets_tsv, "--out", str(active_dir)],
        check=True,
    )

    # 3. residue maps (build_residue_map handles AF: prefix)
    subprocess.run(
        ["python", "-m", "src.data.build_residue_map",
         "--active-sites", str(active_dir),
         "--fastas", str(fasta_dir),
         "--out", str(residue_dir),
         "--sifts-cache", str(sifts_dir),
         "--uniprot-cache", str(unp_dir),
         "--structures", str(struct_dir),
         "--update-active-sites"],
        check=False,  # non-zero exit signals diagnostics, not fatal
    )

    print(f"\n[setup] {args.set_name} ready under {root}")


if __name__ == "__main__":
    sys.exit(main())
