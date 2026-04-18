# Fold Classifier for Embedded Deployment

## Goal

A tiny sequence-in, fold-class-out classifier that runs on a Raspberry Pi and
predicts whether a given **nucleotide** sequence will fold into one of ~5
reference structures (or "none of the above"). Structure awareness is baked
in during training via a folding oracle; inference is pure sequence.

**Input contract (v1):** nucleotide sequence in a known reading frame
(frame 0, length divisible by 3, no stops expected mid-sequence). A
translation layer (codon table lookup) converts to amino acids before the
main model. The translate step is part of the exported artifact so the
deployed model takes nucleotides directly. Internally, everything
downstream of translate operates on AA, identical to an AA-input model.

**Deferred to v2:** ORF recognition / frame detection / handling of
sequences with stops or UTRs. See Future Work.

## Non-goals

- On-device structure prediction (infeasible on a Pi).
- Discovering novel fold determinants. This is distillation of a folding
  oracle into a small model, not biological discovery.
- Beating AlphaFold at anything. We are trading accuracy for footprint.

## Pipeline

1. **Reference selection.** Pick 5 target scaffolds with diverse folds,
   ideally with existing DMS / stability data for sanity checks. Solve
   structures (PDB) required.
2. **Perturbation library.** For each scaffold, generate:
   - All single mutants (saturation).
   - Sampled double / triple mutants.
   - Higher-order perturbations stratified by edit distance (up to ~30% of
     length).
   - Composition-matched scrambles (hard negatives, zero edit-distance
     signal).
   - Decoys: sequences drawn from unrelated folds (PDB).
   - Distant homologs (<30% seq id, same fold) as positives where available.
3. **Oracle labeling.** Fold each sequence with ESMFold. Compute TM-score
   against each of the 5 references. Label = argmax if max TM > 0.5, else
   "none".
4. **Dataset balancing.** Rebalance across (edit-distance bucket, label)
   cells so the trivial "distance from WT" classifier is uninformative. Hard
   negative mining on low-distance misfolders and high-distance correct
   folders.
5. **Student training.** Small 1D CNN or tiny transformer, ~1-5M params.
   Train on soft TM-score targets (regression head per class) or hard
   argmax labels. Int8 quantize.
6. **Evaluation.**
   - Held out perturbation magnitudes.
   - Held out distant homologs of targets.
   - Unrelated-fold decoys.
   - Baselines: BLAST/HMMER against the 5 targets, k-mer SVM,
     nearest-WT edit distance. Student must beat these on low-seq-id cases
     to count as a real result.
7. **Deployment.** Quantize, export to ONNX or TFLite, benchmark on Pi 4/5.
   Target: <10MB, <100ms/seq, <50MB RAM.

## Compute requirements

**Oracle (ESMFold) label generation — the bottleneck.**

- ESMFold VRAM: ~14GB for a 200-residue protein, scales ~quadratically with
  length. For up-to-300 residue targets, want >=24GB VRAM. 16GB is tight.
- Throughput on A100 40GB: ~2-5s per sequence at 200 residues.
- Dataset size estimate: 5 scaffolds * ~50k sequences = 250k folds.
- At 3s/seq on one A100: ~210 GPU-hours. ~9 days on one GPU, <1 day on 8.

**Recommended allocation on ycluster:**

- Preferred: 1x A100 80GB or H100 for headroom on longer targets and
  batching. 2-4 in parallel gets the whole dataset in ~2-3 days.
- Acceptable: A100 40GB, A6000 48GB, or RTX 6000 Ada 48GB. Fine for
  <=250 residue targets.
- Minimum viable: RTX 3090/4090 24GB. Works for short proteins, will OOM
  on anything >~250 residues without chunking tricks.
- Avoid: V100 16GB, anything smaller. Memory pressure will dominate.

**Student training.** Negligible. Any GPU, including a 2080Ti or CPU for
the smallest variants. Expect <1 GPU-hour end-to-end per training run.

**Evaluation folds (for held-out test set).** Small — a few hundred to a
few thousand sequences. Same GPU as labeling.

## Future work (v2+)

- **ORF recognition.** Detect reading frame and start/stop codons from raw
  nucleotide input. Options: a small upstream classifier head, or a
  rule-based scan trying all 6 frames and picking the one with highest
  confidence from the main model.
- **UTR / multi-ORF inputs.** Handle sequences with flanking non-coding
  regions or multiple candidate ORFs.
- **Codon-level signal.** v1 discards synonymous-codon information at the
  translate step. A codon-aware variant could pick up signal from codon
  usage / mRNA structure, which can affect folding kinetics in vivo.

## Risks

- **Oracle mutation-insensitivity.** ESMFold may call almost everything
  correctly folded. Mitigation: check label distribution early, cross-check
  with Boltz / AF2 + MSA-subsampling on a subset, compare to any
  ProteinGym DMS overlap.
- **Sequence-similarity shortcut.** Student may just learn nearest-WT.
  Mitigation: balanced sampling, hard negatives, scrambles, low-seq-id
  eval. This is the main "did it work" question.
- **5 classes is small.** Overfitting risk is real; may need regularization
  or auxiliary objectives (e.g., predict TM-score to each reference as a
  multi-task regression).

## Milestones

1. Single-scaffold prototype end-to-end. One target, ~5k sequences,
   ESMFold labels, tiny CNN, Pi benchmark. Proves the pipeline.
2. Scale to 5 scaffolds with full perturbation library.
3. Hard-negative mining and rebalancing pass.
4. Baselines and held-out evaluation.
5. Quantization, export, Pi deployment.

## Repo layout

- `src/data/` perturbation generation, dataset assembly
- `src/oracle/` ESMFold wrapper, TM-score computation, labeling
- `src/student/` model defs, training, quantization, export
- `src/eval/` baselines, held-out evaluation, Pi benchmarks
- `scripts/` slurm submission, one-off jobs
- `configs/` experiment configs
- `notebooks/` exploration, label QC, figures
