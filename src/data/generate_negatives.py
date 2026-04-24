"""Generate non-MPNN negative classes for each target:

  ala_scan      : single alanine substitutions at each fixed-position residue
                  of the target WT and of each MPNN positive (if available).
  combined_ko   : all fixed-position residues mutated in the same sequence
                  (ala, and separately charge-flip D/E<->K, H->E, R->E).
  scramble      : composition-matched random permutation of the WT sequence.
  perturb30     : random substitutions at ~30% of non-fixed positions (fold
                  likely broken, function definitely broken).

Writes one FASTA per class per target:
  data/variants/<class>/<TID>.fasta

Uses only the target WT and active_sites JSON. Needs MPNN positives only
for "ala_scan" in its "apply to MPNN backbones" mode (optional).
"""
import argparse, json, pathlib, random, sys

from src.data.residue_map import ResidueMap


CHARGE_FLIP = {"D": "K", "E": "K", "K": "D", "R": "E", "H": "E"}


def read_fasta(p):
    name, seq = None, []
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                yield name, "".join(seq)
            name = line[1:]
            seq = []
        else:
            seq.append(line)
    if name is not None:
        yield name, "".join(seq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--primary-seqs", default="data/targets/primary_sequences")
    ap.add_argument("--active-sites", default="data/targets/active_sites")
    ap.add_argument("--residue-maps", default="data/targets/residue_maps",
                    help="per-target SIFTS-backed residue maps "
                         "(built by src.data.build_residue_map)")
    ap.add_argument("--mpnn-fasta", default="data/variants/mpnn_positives")
    ap.add_argument("--out-root", default="data/variants")
    ap.add_argument("--ala-scan-per-wt", type=int, default=50,
                    help="number of ala-scan mutants of WT per target "
                         "(usually capped by number of fixed positions)")
    ap.add_argument("--ala-scan-of-mpnn", type=int, default=100,
                    help="ala-scan against this many random MPNN positives")
    ap.add_argument("--combined-ko-n", type=int, default=200,
                    help="combined-KO variants per target")
    ap.add_argument("--scramble-n", type=int, default=300)
    ap.add_argument("--perturb30-n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tid = args.target
    random.seed(args.seed)

    # ---- load WT and canonical active-site mask -------------------------
    wt_fa = pathlib.Path(args.primary_seqs, f"{tid}.fasta")
    _, wt = next(read_fasta(wt_fa))
    sites = json.load(open(pathlib.Path(args.active_sites, f"{tid}.json")))
    rmap = ResidueMap.load(pathlib.Path(args.residue_maps, f"{tid}.json"))

    fixed_uniprot = sites.get("fixed_positions_uniprot")
    if fixed_uniprot is None:
        raise SystemExit(
            f"{tid}: active_sites missing fixed_positions_uniprot; "
            f"run src.data.build_residue_map --update-active-sites first"
        )

    # WT-fasta indices for the active-site mask (UniProt residues not
    # covered by the WT FASTA are dropped silently; diagnostics from the
    # residue-map build catch those).
    fixed_wt_idx = rmap.fixed_wt_idx(fixed_uniprot)
    # 1-indexed positions in ProteinMPNN-designed sequences (same order
    # as PDB-resolved residues). Used for ala-scan / combined-ko against
    # MPNN backgrounds.
    fixed_mpnn_1idx = rmap.fixed_mpnn_1idx(fixed_uniprot)
    fixed_mpnn_0idx = [k - 1 for k in fixed_mpnn_1idx]
    if len(fixed_wt_idx) == 0:
        print(f"[{tid}] WARNING: no resolvable fixed positions in WT; "
              "negatives will be ill-defined for this target.", file=sys.stderr)

    out_root = pathlib.Path(args.out_root)

    # ---- helpers ---------------------------------------------------------
    def write_fasta(sub, records):
        d = out_root / sub; d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{tid}.fasta", "w") as fh:
            for name, seq in records:
                fh.write(f">{name}\n{seq}\n")

    def apply_muts(seq, muts):
        """muts: iterable of (idx, new_aa)."""
        s = list(seq)
        for i, aa in muts:
            if 0 <= i < len(s) and s[i] != "X":
                s[i] = aa
        return "".join(s)

    # ---- class: ala_scan (WT) -------------------------------------------
    ala_recs = []
    for i in fixed_wt_idx:
        if wt[i] == "X":
            continue
        new = apply_muts(wt, [(i, "A")])
        ala_recs.append((f"{tid}_ala_wt_pos{i+1}_{wt[i]}A", new))
    # optional: ala-scan of MPNN designs (preserve realism of backgrounds).
    # MPNN designs are indexed by PDB-resolved-residue order, so the
    # fixed positions use `fixed_mpnn_0idx` (NOT fixed_wt_idx).
    mpnn_path = pathlib.Path(args.mpnn_fasta, f"{tid}.fasta")
    if mpnn_path.exists() and fixed_mpnn_0idx:
        mpnn_list = list(read_fasta(mpnn_path))
        sample = random.sample(mpnn_list, min(args.ala_scan_of_mpnn, len(mpnn_list)))
        for name, seq in sample:
            i = random.choice(fixed_mpnn_0idx)
            if i >= len(seq) or seq[i] == "X":
                continue
            new = apply_muts(seq, [(i, "A")])
            ala_recs.append((f"{tid}_ala_mpnn_{name.split()[0]}_mpos{i+1}", new))
    write_fasta("ala_scan", ala_recs)

    # ---- class: combined_ko ---------------------------------------------
    combined = []
    # variant 1: ALL fixed positions -> A, WT background
    if fixed_wt_idx:
        new = apply_muts(wt, [(i, "A") for i in fixed_wt_idx])
        combined.append((f"{tid}_ko_all_ala_wt", new))
    # variant 2: charge-flip, WT background
    muts = []
    for i in fixed_wt_idx:
        src = wt[i]
        if src in CHARGE_FLIP:
            muts.append((i, CHARGE_FLIP[src]))
        else:
            muts.append((i, "A"))
    if muts:
        combined.append((f"{tid}_ko_flip_wt", apply_muts(wt, muts)))
    # fill to combined-ko-n with MPNN-background combined KOs.
    # Uses fixed_mpnn_0idx (MPNN-relative indices), not fixed_wt_idx.
    if mpnn_path.exists() and fixed_mpnn_0idx:
        mpnn_list = list(read_fasta(mpnn_path))
        random.shuffle(mpnn_list)
        for name, seq in mpnn_list[:args.combined_ko_n]:
            new = apply_muts(seq, [(i, "A") for i in fixed_mpnn_0idx])
            combined.append((f"{tid}_ko_all_ala_{name.split()[0]}", new))
    write_fasta("combined_ko", combined)

    # ---- class: scramble -------------------------------------------------
    scrambles = []
    wt_chars = list(wt.replace("X", ""))
    for k in range(args.scramble_n):
        random.shuffle(wt_chars)
        scrambles.append((f"{tid}_scramble_{k}", "".join(wt_chars)))
    write_fasta("scramble", scrambles)

    # ---- class: perturb30 ------------------------------------------------
    perturb_recs = []
    aa_alphabet = list("ACDEFGHIKLMNPQRSTVWY")
    non_fixed = [i for i in range(len(wt)) if i not in set(fixed_wt_idx) and wt[i] != "X"]
    n_sub = max(1, int(0.3 * len(non_fixed)))
    for k in range(args.perturb30_n):
        picks = random.sample(non_fixed, n_sub)
        muts = [(i, random.choice(aa_alphabet)) for i in picks]
        new = apply_muts(wt, muts)
        perturb_recs.append((f"{tid}_perturb30_{k}", new))
    write_fasta("perturb30", perturb_recs)

    counts = {
        "ala_scan": len(ala_recs),
        "combined_ko": len(combined),
        "scramble": len(scrambles),
        "perturb30": len(perturb_recs),
    }
    print(json.dumps({"target": tid, "counts": counts,
                      "n_fixed_wt": len(fixed_wt_idx),
                      "n_fixed_mpnn": len(fixed_mpnn_0idx)}))


if __name__ == "__main__":
    sys.exit(main())
