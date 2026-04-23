"""Export a trained CapitiCNN to ONNX + a small metadata JSON.

The ONNX graph takes a (1, max_len) int64 tensor of AA token IDs and
returns a (1, num_classes) float tensor of logits. The CLI does NT->AA
translation and tokenisation in host code before feeding the model.
"""
import argparse, json, pathlib, sys
import torch
from src.student.model import CapitiCNN, AA_TO_IDX, PAD_IDX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/runs/v1/best.pt")
    ap.add_argument("--out-onnx", default="data/runs/v1/capiti.onnx")
    ap.add_argument("--out-meta", default="data/runs/v1/capiti.meta.json")
    ap.add_argument("--max-len", type=int, default=800)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.0)
    args = ap.parse_args()

    model = CapitiCNN(channels=args.channels, dropout=args.dropout)
    sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()

    dummy = torch.zeros((1, args.max_len), dtype=torch.long)
    out_onnx = pathlib.Path(args.out_onnx); out_onnx.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, out_onnx,
        input_names=["token_ids"], output_names=["logits"],
        dynamic_axes={"token_ids": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )

    meta = {
        "max_len": args.max_len,
        "vocab": {"pad": PAD_IDX, **{aa: i for aa, i in AA_TO_IDX.items()}},
        "labels": [f"T{i}" for i in range(1, 10)] + ["none"],
        "none_idx": 9,
    }
    pathlib.Path(args.out_meta).write_text(json.dumps(meta, indent=2))
    print(f"exported {out_onnx} ({out_onnx.stat().st_size/1e6:.2f} MB)")
    print(f"meta -> {args.out_meta}")


if __name__ == "__main__":
    sys.exit(main())
