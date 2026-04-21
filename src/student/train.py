"""Train the CapitiCNN on data/dataset/dataset.tsv.

Balanced sampling by (target, class) cell. Multi-class cross-entropy over
9 targets + "none". At eval we also derive the binary "in-set vs not"
prediction as 1 - p(none).
"""
import argparse, collections, csv, json, pathlib, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from src.student.model import CapitiCNN, encode, PAD_IDX


TARGETS = [f"T{i}" for i in range(1, 10)]
LABELS = TARGETS + ["none"]            # 10 classes
LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}
NONE_IDX = LABEL_TO_IDX["none"]


class CapitiDataset(Dataset):
    def __init__(self, rows, max_len):
        self.rows = rows
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        ids = torch.tensor(encode(r["seq"], self.max_len), dtype=torch.long)
        y = torch.tensor(LABEL_TO_IDX[r["target"]], dtype=torch.long)
        return ids, y


def load_tsv(path):
    out = []
    with open(path) as fh:
        r = csv.DictReader(fh, delimiter="\t")
        for row in r:
            out.append(row)
    return out


def make_balanced_sampler(rows):
    """Weight each (target, class) cell uniformly so the loss doesn't get
       dominated by MPNN positives (9k) vs rare KO classes (hundreds).
    """
    counts = collections.Counter((r["target"], r["class"]) for r in rows)
    w = [1.0 / counts[(r["target"], r["class"])] for r in rows]
    return WeightedRandomSampler(w, num_samples=len(rows), replacement=True)


def metrics(logits, y):
    """logits (N, C), y (N,) -> dict of metrics."""
    p = F.softmax(logits, dim=-1).numpy()
    y = y.numpy()
    pred = p.argmax(axis=1)
    bin_score = 1.0 - p[:, NONE_IDX]
    bin_y = (y != NONE_IDX).astype(int)

    # ROC-AUC (binary)
    order = np.argsort(-bin_score)
    y_sorted = bin_y[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    P, N = bin_y.sum(), (1 - bin_y).sum()
    tpr = tps / max(P, 1)
    fpr = fps / max(N, 1)
    # prepend (0,0)
    tpr = np.concatenate(([0.], tpr)); fpr = np.concatenate(([0.], fpr))
    auc = np.trapz(tpr, fpr)

    # PR-AUC
    prec = tps / (tps + fps).clip(min=1)
    rec = tpr[1:]
    order2 = np.argsort(rec)
    prauc = np.trapz(prec[order2], rec[order2])

    # Accuracy (10-way) and on-positive confusion
    acc10 = (pred == y).mean()
    pos_mask = y != NONE_IDX
    acc_pos = (pred[pos_mask] == y[pos_mask]).mean() if pos_mask.any() else float("nan")
    acc_neg = (pred[~pos_mask] == NONE_IDX).mean() if (~pos_mask).any() else float("nan")
    return dict(binary_auc=float(auc), binary_prauc=float(prauc),
                acc10=float(acc10), acc_on_pos=float(acc_pos),
                acc_on_neg=float(acc_neg))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_y = [], []
    for ids, y in loader:
        ids = ids.to(device); y = y.to(device)
        logits = model(ids)
        all_logits.append(logits.cpu()); all_y.append(y.cpu())
    return metrics(torch.cat(all_logits), torch.cat(all_y))


def per_class_breakdown(model, rows, max_len, device, batch_size=128):
    model.eval()
    # compute per-class accuracy
    ds = CapitiDataset(rows, max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    all_logits = []
    with torch.no_grad():
        for ids, _ in loader:
            all_logits.append(model(ids.to(device)).cpu())
    logits = torch.cat(all_logits)
    p = F.softmax(logits, dim=-1).numpy()
    pred = p.argmax(axis=1)
    bin_score = 1.0 - p[:, NONE_IDX]

    out = {}
    by_cell = collections.defaultdict(list)
    for i, r in enumerate(rows):
        key = r["class"]
        y_true = LABEL_TO_IDX[r["target"]]
        by_cell[key].append((y_true, pred[i], bin_score[i]))
    for cls, entries in by_cell.items():
        ys = np.array([e[0] for e in entries])
        ps = np.array([e[1] for e in entries])
        scores = np.array([e[2] for e in entries])
        is_pos = (ys != NONE_IDX).astype(int)
        n = len(entries)
        acc = float((ps == ys).mean())
        if is_pos.max() == is_pos.min():
            # only one class present -- report score-at-threshold 0.5 as bin_acc
            bin_acc = float(((scores >= 0.5).astype(int) == is_pos).mean())
        else:
            bin_acc = float(((scores >= 0.5).astype(int) == is_pos).mean())
        out[cls] = {"n": n, "acc10": acc, "bin_acc_at_0.5": bin_acc,
                    "mean_inset_score": float(scores.mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/dataset/dataset.tsv")
    ap.add_argument("--out-dir", default="data/runs/v1")
    ap.add_argument("--max-len", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    rows = load_tsv(args.dataset)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows   = [r for r in rows if r["split"] == "val"]
    test_rows  = [r for r in rows if r["split"] == "test"]
    print(f"splits: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    train_ds = CapitiDataset(train_rows, args.max_len)
    val_ds   = CapitiDataset(val_rows, args.max_len)
    test_ds  = CapitiDataset(test_rows, args.max_len)

    sampler = make_balanced_sampler(train_rows)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    model = CapitiCNN(channels=args.channels, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    out = pathlib.Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    history = []
    best_val_auc = -1.0
    for ep in range(args.epochs):
        model.train()
        losses = []
        t0 = time.time()
        for ids, y in train_loader:
            ids = ids.to(device); y = y.to(device)
            logits = model(ids)
            loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        sched.step()
        vm = evaluate(model, val_loader, device)
        rec = dict(epoch=ep, train_loss=float(np.mean(losses)),
                   elapsed_s=round(time.time() - t0, 1), **vm)
        history.append(rec)
        print(json.dumps(rec), flush=True)
        if vm["binary_auc"] > best_val_auc:
            best_val_auc = vm["binary_auc"]
            torch.save(model.state_dict(), out / "best.pt")

    # reload best and test
    model.load_state_dict(torch.load(out / "best.pt"))
    tm = evaluate(model, test_loader, device)
    print("test:", json.dumps(tm))
    cls_break = per_class_breakdown(model, test_rows, args.max_len, device)
    print("per-class test:")
    for k, v in sorted(cls_break.items()):
        print(f"  {k}: {v}")

    (out / "history.json").write_text(json.dumps(history, indent=2))
    (out / "test_metrics.json").write_text(
        json.dumps(dict(overall=tm, per_class=cls_break), indent=2))
    print(f"\nbest val binary AUC: {best_val_auc:.4f}")


if __name__ == "__main__":
    sys.exit(main())
