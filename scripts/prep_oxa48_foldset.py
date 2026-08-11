"""Build the ESMFold input set for the OXA-48 (T3) poster figure.

Extracts the wild-type chain A sequence + a clean single-chain reference PDB
from the experimental structure, samples 15 ProteinMPNN designs evenly across
the seq_recovery range, and writes 16 single-record FASTAs (one per sequence,
as fold_esmfold.py expects).
"""
import pathlib, re

DATA = pathlib.Path("/nfs/roberts/scratch/pi_skr2/mcn26/capiti/data")
PDB_IN = DATA / "targets/structures/T3.pdb"
DESIGNS = DATA / "variants/mpnn_positives/T3.fasta"
OUT = DATA / "oracle_folds/oxa48_poster"
FASTA_DIR = OUT / "inputs"
N_DESIGNS = 15

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "KCX": "K",  # carboxylated Lys73 -> standard Lys for folding
}


def wt_from_pdb(path, chain="A"):
    """Sequence + chain-A ATOM lines from the first copy in the tetramer."""
    seq, atom_lines, seen = [], [], set()
    for line in path.read_text().splitlines():
        if line[:6] not in ("ATOM  ", "HETATM"):
            continue
        if line[21] != chain:
            continue
        resname, resnum = line[17:20].strip(), line[22:27]
        if line[:6] == "HETATM" and resname not in THREE_TO_ONE:
            continue  # skip EDO / HOH, keep KCX
        atom_lines.append(line)
        if resnum not in seen:
            seen.add(resnum)
            seq.append(THREE_TO_ONE.get(resname, "X"))
    return "".join(seq), atom_lines


def read_designs(path):
    recs = []
    name = seq = rec_id = None
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                recs.append((name, rec_id, "".join(seq)))
            name = line[1:].split()[0]
            m = re.search(r"seq_recovery=([0-9.]+)", line)
            rec_id = float(m.group(1)) if m else None
            seq = []
        elif line.strip():
            seq.append(line.strip())
    if name is not None:
        recs.append((name, rec_id, "".join(seq)))
    return recs


def main():
    FASTA_DIR.mkdir(parents=True, exist_ok=True)

    wt_seq, wt_atoms = wt_from_pdb(PDB_IN)
    (OUT / "T3_WT_ref_chainA.pdb").write_text(
        "\n".join(wt_atoms) + "\nEND\n")
    (FASTA_DIR / "T3_WT.fasta").write_text(f">T3_WT\n{wt_seq}\n")
    print(f"WT chain A: {len(wt_seq)} residues")

    designs = read_designs(DESIGNS)
    designs.sort(key=lambda r: r[1])  # by seq_recovery, most divergent first
    # evenly spaced indices across the sorted range, inclusive of both ends
    idx = [round(i * (len(designs) - 1) / (N_DESIGNS - 1))
           for i in range(N_DESIGNS)]
    picked = [designs[i] for i in idx]
    for name, rec, seq in picked:
        (FASTA_DIR / f"{name}.fasta").write_text(f">{name}\n{seq}\n")
    print(f"sampled {len(picked)} designs, "
          f"seq_recovery {picked[0][1]:.3f}..{picked[-1][1]:.3f}")
    print(f"inputs written to {FASTA_DIR}")


if __name__ == "__main__":
    main()
