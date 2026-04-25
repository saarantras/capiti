"""Generate ProteinMPNN variants for a single target with active-site
residues fixed. One target per invocation (so this fits a slurm array).

Reads:
  data/targets/structures/<TID>.pdb
  data/targets/active_sites/<TID>.json
Writes:
  data/variants/mpnn_positives/<TID>.fasta   (consolidated, one design per entry)

Each output FASTA record:
  >TID_mpnn_<temp>_<idx> score=... seq_recovery=... source=mpnn
  <sequence>

Conversion: the MPNN PDB parser fills gaps in chain A with X and emits a
sequence of length (max_author - min_author + 1). Our active-site residues
are recorded in PDB author numbering; the 1-indexed MPNN position is
therefore (author - min_author + 1).
"""
import argparse, json, os, pathlib, shutil, subprocess, sys, tempfile


def chain_author_range(pdb_path, chain="A"):
    lo, hi = None, None
    for line in open(pdb_path):
        if line.startswith("ATOM") and line[21] == chain:
            r = int(line[22:26])
            lo = r if lo is None else min(lo, r)
            hi = r if hi is None else max(hi, r)
    return lo, hi


# back-compat alias for older callers
chain_a_author_range = chain_author_range


def run(cmd, env=None):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        sys.stderr.write(p.stdout)
        sys.stderr.write(p.stderr)
        raise SystemExit(f"command failed: {cmd}")
    return p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="e.g. T2")
    ap.add_argument("--mpnn-root", default="third_party/ProteinMPNN")
    ap.add_argument("--structures", default="data/targets/structures")
    ap.add_argument("--active-sites", default="data/targets/active_sites")
    ap.add_argument("--residue-maps", default="data/targets/residue_maps")
    ap.add_argument("--out-dir", default="data/variants/mpnn_positives")
    ap.add_argument("--work-dir", default=None,
                    help="MPNN scratch directory (default: sibling of "
                         "--out-dir called _mpnn_work)")
    ap.add_argument("--num-per-temp", type=int, default=500)
    ap.add_argument("--temps", nargs="+", default=["0.1", "0.3"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-name", default="v_48_020",
                    help="soluble model noise level")
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    tid = args.target
    mpnn = pathlib.Path(args.mpnn_root).resolve()
    pdb = pathlib.Path(args.structures, f"{tid}.pdb").resolve()
    sites = json.load(open(pathlib.Path(args.active_sites, f"{tid}.json")))
    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    chain = sites.get("chain", "A")
    lo, hi = chain_author_range(pdb, chain)
    print(f"[{tid}] chain {chain} author range {lo}-{hi}")

    # Canonical source of truth: UniProt residue numbers + SIFTS cache.
    from src.data.residue_map import ResidueMap
    rmap = ResidueMap.load(pathlib.Path(args.residue_maps, f"{tid}.json"))
    fixed_uniprot = sites.get("fixed_positions_uniprot")
    if fixed_uniprot is None:
        raise SystemExit(
            f"{tid}: active_sites missing fixed_positions_uniprot; "
            f"run src.data.build_residue_map --update-active-sites first"
        )
    fixed_mpnn = rmap.fixed_mpnn_1idx(fixed_uniprot)
    print(f"[{tid}] fixed uniprot={fixed_uniprot}")
    print(f"[{tid}] fixed mpnn (1-indexed)={fixed_mpnn}")

    # keep work dir visible under data/ for debuggability
    work_base = pathlib.Path(args.work_dir) if args.work_dir else \
        pathlib.Path(args.out_dir).parent / "_mpnn_work"
    work_base.mkdir(parents=True, exist_ok=True)
    td = work_base / tid
    if td.exists():
        shutil.rmtree(td)
    td.mkdir()
    if True:
        # MPNN's parser reads a directory of PDBs; stage just this target
        stage = td / "pdbs"; stage.mkdir()
        shutil.copy(pdb, stage / f"{tid}.pdb")

        parsed = td / "parsed.jsonl"
        chains = td / "chains.jsonl"
        fixed = td / "fixed.jsonl"
        run(f"python {mpnn}/helper_scripts/parse_multiple_chains.py "
            f"--input_path={stage} --output_path={parsed}")
        run(f"python {mpnn}/helper_scripts/assign_fixed_chains.py "
            f"--input_path={parsed} --output_path={chains} --chain_list='{chain}'")
        pos_str = " ".join(str(i) for i in fixed_mpnn) or "0"  # MPNN wants at least one token
        run(f"python {mpnn}/helper_scripts/make_fixed_positions_dict.py "
            f"--input_path={parsed} --output_path={fixed} "
            f"--chain_list='{chain}' --position_list='{pos_str}'")

        mpnn_out = td / "out"; mpnn_out.mkdir()
        for temp in args.temps:
            run(f"python {mpnn}/protein_mpnn_run.py "
                f"--jsonl_path {parsed} "
                f"--chain_id_jsonl {chains} "
                f"--fixed_positions_jsonl {fixed} "
                f"--out_folder {mpnn_out / temp} "
                f"--num_seq_per_target {args.num_per_temp} "
                f"--sampling_temp {temp} "
                f"--batch_size {args.batch_size} "
                f"--path_to_model_weights {mpnn}/soluble_model_weights "
                f"--model_name {args.model_name} "
                f"--seed {args.seed}")

        # consolidate outputs -> one FASTA per target
        out_fa = out_dir / f"{tid}.fasta"
        with open(out_fa, "w") as fh:
            for temp in args.temps:
                seqs_dir = mpnn_out / temp / "seqs"
                # MPNN writes <pdb_stem>.fa; there should be exactly one
                fa = next(seqs_dir.glob("*.fa"))
                lines = open(fa).read().splitlines()
                # first entry is the WT echo; skip it
                i = 0
                idx = 0
                while i < len(lines):
                    if lines[i].startswith(">"):
                        hdr, seq = lines[i], lines[i + 1]
                        i += 2
                        if "sample=" not in hdr:
                            continue  # skip WT echo
                        # re-header
                        score = None; rec = None
                        for part in hdr.split(","):
                            part = part.strip()
                            if part.startswith("score="):
                                score = part.split("=")[1]
                            if part.startswith("seq_recovery="):
                                rec = part.split("=")[1]
                        fh.write(f">{tid}_mpnn_t{temp}_{idx} "
                                 f"score={score} seq_recovery={rec} source=mpnn\n{seq}\n")
                        idx += 1
                    else:
                        i += 1
        # simple count report
        n = sum(1 for l in open(out_fa) if l.startswith(">"))
        print(f"[{tid}] wrote {n} designs -> {out_fa}")


if __name__ == "__main__":
    sys.exit(main())
