"""Render a text summary and a graph diagram of CapitiCNN."""
import argparse, pathlib
import torch
from torchinfo import summary
from torchview import draw_graph

from src.student.model import CapitiCNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=800)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--out-dir", default="docs")
    ap.add_argument("--fmt", default="svg", choices=["svg", "png", "pdf"])
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model = CapitiCNN(channels=args.channels, dropout=0.0).eval()

    txt = summary(
        model,
        input_data=torch.zeros((1, args.max_len), dtype=torch.long),
        depth=4,
        col_names=("input_size", "output_size", "num_params", "mult_adds"),
        verbose=0,
    )
    (out_dir / "capiti.summary.txt").write_text(str(txt))
    print(f"summary -> {out_dir / 'capiti.summary.txt'}")

    g = draw_graph(
        model,
        input_data=torch.zeros((1, args.max_len), dtype=torch.long),
        graph_name="CapitiCNN",
        depth=3,
        expand_nested=True,
    )
    g.visual_graph.render(
        filename=str(out_dir / "capiti.arch"),
        format=args.fmt,
        cleanup=True,
    )
    print(f"diagram -> {out_dir / f'capiti.arch.{args.fmt}'}")


if __name__ == "__main__":
    main()
