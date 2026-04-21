"""Fold one or more FASTA sequences with ESMFold and write PDB + metadata.

Usage:
    python fold_esmfold.py --fasta path.fasta [path2.fasta ...] --out out_dir

Writes <name>.pdb and appends one JSON record per sequence to manifest.jsonl
(len, plddt_mean, plddt_median, time_s, gpu_mem_gb).
"""
import argparse, json, time, pathlib, sys
import torch
from transformers import AutoTokenizer, EsmForProteinFolding


def read_fasta(path):
    name, seq = None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                name = line[1:].split()[0]
            else:
                seq.append(line)
    return name, "".join(seq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk-size", type=int, default=64,
                    help="axial attention chunk size; lower = less VRAM, slower")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)

    tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1", torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    model = model.to(device)
    model.esm = model.esm.half() if device == "cuda" else model.esm
    model.trunk.set_chunk_size(args.chunk_size)
    model.eval()

    for f in args.fasta:
        name, seq = read_fasta(f)
        print(f"folding {name} len={len(seq)}", flush=True)
        torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
        t0 = time.time()
        with torch.no_grad():
            toks = tok([seq], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
            out_obj = model(toks)
            pdb_str = model.output_to_pdb(out_obj)[0]
        dt = time.time() - t0
        plddt = out_obj["plddt"].float().cpu().numpy()[0]
        mask = plddt.sum(axis=-1) > 0
        per_res = plddt[mask].mean(axis=-1)
        peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
        (out / f"{name}.pdb").write_text(pdb_str)
        rec = dict(name=name, length=len(seq), time_s=round(dt, 2),
                   plddt_mean=float(per_res.mean()), plddt_median=float(sorted(per_res)[len(per_res)//2]),
                   peak_gpu_gb=round(peak_gb, 2))
        print(json.dumps(rec), flush=True)
        with open(manifest, "a") as mh:
            mh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    sys.exit(main())
