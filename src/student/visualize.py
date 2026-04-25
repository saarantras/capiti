"""Render a text summary and two-level graph diagrams of CapitiCNN:

  capiti.overview.<fmt> : top-level view with the 5 ResidualDilatedBlocks
                          collapsed into a single "x N" box. Hand-built via
                          graphviz, because torchview won't collapse
                          identical siblings.
  capiti.block.<fmt>    : what a single ResidualDilatedBlock contains.
  capiti.summary.txt    : torchinfo layer table (sizes, params, mult-adds).
"""
import argparse, pathlib
import torch
from torchinfo import summary
from torchview import draw_graph
import graphviz

from src.student.model import CapitiCNN, ResidualDilatedBlock


def build_overview(channels, embed_dim, num_blocks, dilations, num_classes,
                    max_len, pool, fmt="svg"):
    """Hand-built overview: identical blocks collapsed into one box with
       '(x N)' and the full dilation schedule listed inside.
    """
    dot = graphviz.Digraph("CapitiCNN", format=fmt)
    dot.attr(rankdir="TB", nodesep="0.35", ranksep="0.45")
    dot.attr("node", shape="box", style="rounded,filled",
             fillcolor="#ffffff", fontname="Helvetica", fontsize="10")

    if pool == "mean_max":
        pool_label = "Masked Mean + Max Pool\nover length (concat)"
        head_in = channels * 2
    else:
        pool_label = "Masked Mean Pool\nover length"
        head_in = channels

    nodes = [
        ("in",    f"token_ids\n(B, {max_len})",
                  "#eef6ff"),
        ("embed", f"Embedding\n22 -> {embed_dim}",
                  "#ffffff"),
        ("stem",  f"Conv1d stem\n{embed_dim} -> {channels}, k=1",
                  "#ffffff"),
        ("blk",   f"ResidualDilatedBlock  (x {num_blocks})\n"
                  f"channels={channels}, k=5\n"
                  f"dilations={list(dilations)}",
                  "#fff6d6"),
        ("pool",  pool_label,
                  "#ffffff"),
        ("head",  f"MLP head\n{head_in} -> 128 -> {num_classes}",
                  "#ffffff"),
        ("out",   f"logits\n(B, {num_classes})",
                  "#eef6ff"),
    ]
    for nid, label, fill in nodes:
        dot.node(nid, label=label, fillcolor=fill)
    for (a, _, _), (b, _, _) in zip(nodes, nodes[1:]):
        dot.edge(a, b)
    return dot


def render_torchview(model, out_path, input_data, depth, expand_nested, fmt):
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
    ap.add_argument("--max-len", type=int, default=1200)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--embed-dim", type=int, default=32)
    ap.add_argument("--num-blocks", type=int, default=5)
    ap.add_argument("--dilations", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    ap.add_argument("--num-classes", type=int, default=10)
    ap.add_argument("--out-dir", default="docs")
    ap.add_argument("--fmt", default="svg", choices=["svg", "png", "pdf"])
    ap.add_argument("--pool", choices=("mean", "mean_max"), default="mean_max")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model = CapitiCNN(channels=args.channels, dropout=0.0,
                       pool=args.pool).eval()

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

    # overview (hand-built, blocks collapsed with "x N")
    overview = build_overview(
        channels=args.channels,
        embed_dim=args.embed_dim,
        num_blocks=args.num_blocks,
        dilations=args.dilations,
        num_classes=args.num_classes,
        max_len=args.max_len,
        pool=args.pool,
        fmt=args.fmt,
    )
    overview_path = out_dir / "capiti.overview"
    overview.render(filename=str(overview_path), cleanup=True)
    print(f"overview -> {overview_path.with_suffix('.' + args.fmt)}")

    # block detail: expand internals of one ResidualDilatedBlock
    block = ResidualDilatedBlock(args.channels, kernel_size=5, dilation=4).eval()
    block_out = render_torchview(
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
