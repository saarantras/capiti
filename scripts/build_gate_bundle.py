"""Build a flat gate.json for one bundled set, ready to ship with the
ONNX. The CLI loads this at inference and applies the fixed-position
gate without needing scipy / sklearn / the residue-map cache.

Schema:
  {
    "<target>": {
      "triples": [[wt_idx, mpnn_0idx, expected_aa], ...],
      "wt_len":   int,
      "mpnn_len": int
    },
    ...
  }

Run for each set:
  python scripts/build_gate_bundle.py --set ab9
  python scripts/build_gate_bundle.py --set C
  python scripts/build_gate_bundle.py --set E

Default output path: capiti/_model/<set>/gate.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.data.residue_map import ResidueMap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="set_name", required=True)
    ap.add_argument("--active-sites", default=None,
                    help="default: data/<set>/targets/active_sites for "
                         "non-ab9 sets, data/targets/active_sites for ab9")
    ap.add_argument("--residue-maps", default=None,
                    help="default: matching active-sites root")
    ap.add_argument("--out", default=None,
                    help="default: capiti/_model/<set>/gate.json")
    args = ap.parse_args()

    if args.set_name == "ab9":
        root = Path("data/targets")
    else:
        root = Path(f"data/{args.set_name}/targets")
    active_dir = Path(args.active_sites) if args.active_sites \
        else root / "active_sites"
    rmap_dir = Path(args.residue_maps) if args.residue_maps \
        else root / "residue_maps"
    out = Path(args.out) if args.out \
        else Path(f"capiti/_model/{args.set_name}/gate.json")

    bundle = {}
    n_with_triples = 0
    for asp in sorted(active_dir.glob("*.json")):
        tid = asp.stem
        rmp = rmap_dir / f"{tid}.json"
        if not rmp.exists():
            continue
        active = json.loads(asp.read_text())
        fup = active.get("fixed_positions_uniprot", [])
        if not fup:
            bundle[tid] = {"triples": [],
                            "wt_len": 0, "mpnn_len": 0}
            continue
        rmap = ResidueMap.load(rmp)
        triples = rmap.expected_for_gate(fup)
        bundle[tid] = {
            "triples": [[wi, mi, exp] for wi, mi, exp in triples],
            "wt_len": rmap.wt_length,
            "mpnn_len": rmap.data.get("mpnn_length", 0),
        }
        if triples:
            n_with_triples += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle, indent=2))
    print(f"set={args.set_name}: {len(bundle)} targets, "
          f"{n_with_triples} with triples -> {out} "
          f"({out.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    sys.exit(main())
