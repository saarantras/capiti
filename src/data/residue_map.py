"""Per-target residue-coordinate map, backed by SIFTS.

UniProt numbering is authoritative. For each UniProt residue of the target,
the map records:
  - uniprot_num   (1-indexed, canonical)
  - wt_idx        (0-indexed into the WT FASTA), or None if the WT FASTA
                  doesn't cover this position
  - pdb_num       (PDB author residue number), or None if not resolved
  - pdb_icode     (insertion code, "" if none)
  - mpnn_1idx     (1-indexed into a ProteinMPNN design, which has one
                  residue per resolved PDB CA in order), or None
  - aa            (one-letter AA; UniProt is the source of truth)

Build with `python -m src.data.build_residue_map`. Load with
`ResidueMap.load(path)` and use the lookup helpers for all coordinate
conversions. No other module should do coordinate arithmetic.
"""

from __future__ import annotations

import json
from pathlib import Path


class ResidueMap:
    def __init__(self, data):
        self.data = data
        # index by uniprot_num for O(1) lookups
        self._by_unp = {r["uniprot_num"]: r for r in data["residues"]}
        self._by_pdb = {}
        for r in data["residues"]:
            if r["pdb_num"] is not None:
                self._by_pdb[(r["pdb_num"], r["pdb_icode"])] = r

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    # ----- lookups -----

    @property
    def target(self):
        return self.data["target"]

    @property
    def wt_length(self):
        return self.data["wt_length"]

    def by_uniprot(self, unp_num):
        return self._by_unp.get(unp_num)

    def by_pdb(self, pdb_num, icode=""):
        return self._by_pdb.get((pdb_num, icode))

    def wt_idx(self, unp_num):
        r = self._by_unp.get(unp_num)
        return None if r is None else r["wt_idx"]

    def pdb_num(self, unp_num):
        r = self._by_unp.get(unp_num)
        return None if r is None else r["pdb_num"]

    def mpnn_1idx(self, unp_num):
        r = self._by_unp.get(unp_num)
        return None if r is None else r["mpnn_1idx"]

    def uniprot_for_pdb(self, pdb_num, icode=""):
        r = self._by_pdb.get((pdb_num, icode))
        return None if r is None else r["uniprot_num"]

    def fixed_wt_idx(self, uniprot_positions):
        """Return sorted set of WT-fasta indices for the given list of
        UniProt residue numbers. Skips any position not represented in
        the WT FASTA (e.g., UniProt residues outside the construct)."""
        out = set()
        for n in uniprot_positions:
            i = self.wt_idx(n)
            if i is not None:
                out.add(i)
        return sorted(out)

    def fixed_mpnn_1idx(self, uniprot_positions):
        """Return list of 1-indexed MPNN positions corresponding to the
        given UniProt residue numbers. Skips unresolved residues."""
        out = []
        for n in uniprot_positions:
            k = self.mpnn_1idx(n)
            if k is not None:
                out.append(k)
        return sorted(set(out))

    def expected_for_gate(self, uniprot_positions):
        """Return (wt_idx, mpnn_0idx, expected_aa) triples, one per
        UniProt position that is resolved in BOTH coordinate systems.
        The gate relies on "preserved if either index matches", so
        asymmetric positions (resolvable in WT only, e.g. a
        non-standard residue in the PDB stored as HETATM and invisible
        to ProteinMPNN) would always fire the gate on MPNN-background
        queries. Dropping those positions loses some active-site
        coverage but makes the gate WT/MPNN symmetric by construction;
        combined_ko typically catches the same variants anyway."""
        out = []
        for U in uniprot_positions:
            r = self._by_unp.get(U)
            if r is None:
                continue
            wi = r["wt_idx"]
            mk = r["mpnn_1idx"]
            mi = None if mk is None else mk - 1
            if wi is None or mi is None:
                continue
            out.append((wi, mi, r["aa"]))
        return out
