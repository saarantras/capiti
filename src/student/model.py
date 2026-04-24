"""Small dilated 1D CNN for protein function classification.

Input: (B, L) integer token IDs, where 0 = PAD, 1..20 = amino acids,
       21 = X (unresolved). Output: (B, C) logits for C classes
       (default: 9 targets + "none" = 10).

Designed for Pi deployment: ~500k params, ~65M MACs at L=800, int8-friendly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# Canonical AA ordering: 0 reserved for PAD.
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")  # 20 standard
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AA_LIST)}
AA_TO_IDX["X"] = 21
PAD_IDX = 0
VOCAB_SIZE = 22  # PAD + 20 AA + X


def encode(seq: str, max_len: int) -> list[int]:
    out = [AA_TO_IDX.get(c, AA_TO_IDX["X"]) for c in seq[:max_len]]
    out += [PAD_IDX] * (max_len - len(out))
    return out


class ResidualDilatedBlock(nn.Module):
    def __init__(self, channels, kernel_size=5, dilation=1, dropout=0.1):
        super().__init__()
        pad = ((kernel_size - 1) * dilation) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=pad, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=pad, dilation=dilation)
        self.norm2 = nn.GroupNorm(8, channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # x: (B, C, L)
        r = x
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.drop(x)
        x = self.norm2(self.conv2(x))
        return F.gelu(x + r)


class CapitiCNN(nn.Module):
    def __init__(self, vocab=VOCAB_SIZE, embed_dim=32, channels=64,
                 num_blocks=5, kernel_size=5, num_classes=10,
                 dropout=0.1, dilations=(1, 2, 4, 8, 16),
                 pool="mean", use_aux=False):
        super().__init__()
        assert len(dilations) == num_blocks
        assert pool in ("mean", "mean_max")
        self.pool_mode = pool
        self.embed = nn.Embedding(vocab, embed_dim, padding_idx=PAD_IDX)
        self.stem = nn.Conv1d(embed_dim, channels, kernel_size=1)
        self.blocks = nn.ModuleList([
            ResidualDilatedBlock(channels, kernel_size, dilation=d,
                                  dropout=dropout)
            for d in dilations
        ])
        head_in = channels * (2 if pool == "mean_max" else 1)
        self.head = nn.Sequential(
            nn.Linear(head_in, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        # optional per-residue "is a fixed-position residue" head, used
        # only at training time as an auxiliary loss. Exported inference
        # models should be built with use_aux=False so the extra weights
        # don't ship.
        self.aux_head = (nn.Conv1d(channels, 1, kernel_size=1)
                         if use_aux else None)

    def _trunk(self, ids):
        mask = (ids != PAD_IDX).float()               # (B, L)
        x = self.embed(ids)                           # (B, L, E)
        x = x.transpose(1, 2)                         # (B, E, L)
        x = self.stem(x)                              # (B, C, L)
        for blk in self.blocks:
            x = blk(x)
        return x, mask

    def _pool(self, x, mask):
        mask_c = mask.unsqueeze(1)                    # (B, 1, L)
        denom = mask_c.sum(dim=-1).clamp(min=1.0)
        mean = (x * mask_c).sum(dim=-1) / denom
        if self.pool_mode == "mean":
            return mean
        # mean_max: mask padding to -inf before max
        very_neg = torch.finfo(x.dtype).min
        x_masked = x.masked_fill(mask_c == 0, very_neg)
        mx = x_masked.max(dim=-1).values
        return torch.cat([mean, mx], dim=-1)

    def forward(self, ids):  # ids: (B, L)
        x, mask = self._trunk(ids)
        return self.head(self._pool(x, mask))

    def forward_with_aux(self, ids):
        """Training-only: returns (main_logits (B,C), aux_logits (B,L))."""
        x, mask = self._trunk(ids)
        main = self.head(self._pool(x, mask))
        aux = self.aux_head(x).squeeze(1) if self.aux_head is not None \
            else None
        return main, aux, mask


if __name__ == "__main__":
    m = CapitiCNN()
    n = sum(p.numel() for p in m.parameters())
    print(f"params: {n:,}  ({n * 4 / 1e6:.2f} MB fp32, {n / 1e6:.2f} MB int8)")
    x = torch.randint(1, 22, (2, 800))
    print("output shape:", m(x).shape)
