# Streaming inference: prefix-truncation augmentation

## Motivation

`capiti-listen` reconstructs a sequence one base at a time as the
synthesizer runs. The natural product feature is real-time
classification: as soon as the model is confident the strand being made
is in-class (or definitively isn't), surface that to the user — ideally
with enough headroom to abort a synthesis early via
`capiti-interrupt`.

The current student (`src/student/model.py` → bundled
`capiti/_model/<set>/capiti.onnx`) was trained on full-length, padded
sequences. It will *run* on a partial input — `encode()` pads to
`max_len` regardless — but its score is uncalibrated for any prefix
shorter than the typical training length. C-terminal active-site
residues that the model has learned to depend on are simply absent,
which inflates both false positives and false negatives in
hard-to-predict ways.

The goal of this retrain: produce a student that emits *length-aware,
calibrated* probabilities for any prefix of the input, so a streaming
wrapper on the Pi can apply a sensible `threshold(length)` and stability
gate.

## Approach: prefix-truncation augmentation

Augment each training example by randomly truncating the encoded
sequence to a prefix during `__getitem__`. The model sees both
full-length and partial inputs in roughly the proportions they will
appear at inference time.

### Concretely

In `src/student/train.py`, `CapitiDataset.__getitem__` (around line 105)
currently does:

```python
def __getitem__(self, i):
    r = self.rows[i]
    ids = torch.tensor(encode(r["seq"], self.max_len), dtype=torch.long)
    y = torch.tensor(self.label_to_idx[r["target"]], dtype=torch.long)
    ...
```

Add a prefix-truncation hook gated on a constructor flag so
val/test stay deterministic and full-length:

```python
class CapitiDataset(Dataset):
    def __init__(self, rows, max_len, label_to_idx,
                 per_target_masks=None,
                 prefix_aug: bool = False,
                 prefix_min_frac: float = 0.40,
                 prefix_keep_full: float = 0.50):
        ...
        self.prefix_aug = prefix_aug
        self.prefix_min_frac = prefix_min_frac
        self.prefix_keep_full = prefix_keep_full
        ...

    def __getitem__(self, i):
        r = self.rows[i]
        seq = r["seq"]
        if self.prefix_aug and torch.rand(()).item() > self.prefix_keep_full:
            L = len(seq)
            lo = max(1, int(L * self.prefix_min_frac))
            keep = int(torch.randint(lo, L + 1, ()).item())
            seq = seq[:keep]
        ids = torch.tensor(encode(seq, self.max_len), dtype=torch.long)
        ...
```

Pass `prefix_aug=True` only on the train split (line 260); keep val/test
full-length so headline metrics remain comparable across runs.

### Hyperparameters to sweep

| knob                  | suggested grid       | what it controls                                  |
|-----------------------|----------------------|---------------------------------------------------|
| `prefix_keep_full`    | 0.3, 0.5, 0.7        | fraction of training steps that see the full seq  |
| `prefix_min_frac`     | 0.20, 0.40, 0.60     | shortest prefix we ever show the model            |
| sampling distribution | uniform, beta(2,2)   | bias toward later prefixes if FPR is the worry    |

Start with `prefix_keep_full=0.5`, `prefix_min_frac=0.40`, uniform.

### Aux loss interaction

`build_per_target_masks` (train.py:38) and `forward_with_aux`
(referenced at train.py:303) compute an auxiliary residue-level loss
over the *full* per-target mask. Under prefix truncation the masked
positions past `keep` are now padding. The cleanest fix is to multiply
the existing `pad_mask` (train.py:306) by an additional "is within the
sampled prefix" mask before the aux loss is computed. Concretely, return
the `keep` length from `__getitem__` and trim the aux mask in the train
loop. Don't change the aux mask itself — keep it tied to the full
sequence so it remains valid when no truncation fires.

## What to evaluate after the retrain

A new figure for `docs/benchmark/<run>/`: FPR and TPR as a function of
prefix length percentile (0.4, 0.5, ..., 1.0), per class. The shape of
those curves is what the on-device streaming wrapper will calibrate
against — pick `threshold(length)` so per-bin FPR matches a target
budget (e.g. 1%), and pick the stability gate `K` so the worst-case
single-step FPR drops to your acceptable level.

Add to `src/eval/benchmark.py`: a `--prefix-frac` flag that truncates
each test sequence to `prefix_frac * len(seq)` before scoring, and a
sweep target in the makefile/runner that calls it at 0.4, 0.5, ..., 1.0
and writes the scores into one TSV.

## Out of scope here

- Test-time stability gating logic (`K` consecutive agreeing predictions)
  — that lives in the streaming wrapper on the Pi side, not in
  training.
- Reverse-complement / frame-shift robustness — untouched by this
  change; out-of-frame inputs are the ORF/translate stage's
  responsibility.
- Re-bundling weights into `capiti/_model/<set>/`. After the retrain,
  follow the existing release flow.
