"""Render a text summary and two-level graph diagrams of CapitiCNN:

  capiti.overview.<fmt> : top-level view with each ResidualDilatedBlock
                          shown as a single opaque box.
  capiti.block.<fmt>    : what a single ResidualDilatedBlock contains.
  capiti.summary.txt    : torchinfo layer table (sizes, params, mult-adds).
"""
import argparse, pathlib
import torch
from torchinfo import summary
from torchview import draw_graph

from src.student.model import CapitiCNN, ResidualDilatedBlock


def render(model, out_path, input_data, depth, expand_nested, fmt):
    g = draw_graph(
        model,
        input_data=input_data,
        graph_name=out_path.name,
        depth=depth,
        expand_nested=expand_nested,
    )
    g.visual_graph.render(
        filename=str(out_path),
        format=fmt,
        cleanup=True,
    )
    return out_path.with_suffix(f".{fmt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=800)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--out-dir", default="docs")
    ap.add_argument("--fmt", default="svg", choices=["svg", "png", "pdf"])
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model = CapitiCNN(channels=args.channels, dropout=0.0).eval()

    # text summary
    txt = summary(
        model,
        input_data=torch.zeros((1, args.max_len), dtype=torch.long),
        depth=4,
        col_names=("input_size", "output_size", "num_params", "mult_adds"),
        verbose=0,
    )
    (out_dir / "capiti.summary.txt").write_text(str(txt))
    print(f"summary  -> {out_dir / 'capiti.summary.txt'}")

    # overview: collapse each residual block into a single node
    overview = render(
        model,
        out_dir / "capiti.overview",
        input_data=torch.zeros((1, args.max_len), dtype=torch.long),
        depth=1,
        expand_nested=False,
        fmt=args.fmt,
    )
    print(f"overview -> {overview}")

    # block detail: expand internals of one ResidualDilatedBlock
    block = ResidualDilatedBlock(args.channels, kernel_size=5, dilation=4).eval()
    block_out = render(
        block,
        out_dir / "capiti.block",
        input_data=torch.zeros((1, args.channels, args.max_len), dtype=torch.float32),
        depth=3,
        expand_nested=True,
        fmt=args.fmt,
    )
    print(f"block    -> {block_out}")


if __name__ == "__main__":
    main()
