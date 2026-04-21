# Function Classifier for Embedded Deployment

## Goal

A tiny sequence-in, function-flag-out classifier that runs on a Raspberry Pi.

**Primary task (binary):** given a nucleotide sequence, flag whether it is
expected to retain the enzymatic function of *any* of ~9 reference enzymes
of interest. In-set vs not-in-set is the deployed decision.

**Secondary task (multiclass, free bonus):** when positive, report which of
the 9 it matches. Inter-target confusions (e.g. between two intra-family
siblings) do not hurt the primary task, so multiclass accuracy is reported
but not optimised against.

Function awareness is baked in at dataset construction time: positives come
from design-time function-preserving generators (ProteinMPNN with fixed
active-site residues), negatives come from active-site knockouts, scrambles,
heavy perturbations, unrelated-fold decoys, and broad same-family homologs
that share fold but not specific function (hard negatives for the in-set
decision boundary). A structural oracle (ESMFold) is used only for quality
control on a subsample, not as the primary label source.

**Function definition (strict).** "Function of Ti" means Ti's *specific*
catalytic activity (this enzyme, its substrate preference). A different
beta-lactamase is negative; a class A beta-lactamase homolog that is not
one of T1..T9 is "not in set".

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
- Ground-truth wet-lab function. We are distilling the *design prior* of
  ProteinMPNN + curated active-site annotations, not validated enzymology.
  A positive label means "function is likely preserved by construction",
  not "measured active in a wet lab".
- Beating AlphaFold at anything. We are trading accuracy for footprint.

## Pipeline

1. **Reference selection.** 9 soluble enzyme scaffolds, see
   `configs/targets.tsv`. Structures (PDB) and sequences in
   `data/targets/`.
2. **Active-site annotation.** For each target, collect a fixed-position
   mask: UniProt `ACT_SITE` / `BINDING` / `METAL` / `DISULFID` features
   via SIFTS mapping, with manual supplementation from PDB ligand
   coordination where UniProt is sparse (e.g., T4 zinc-site from Cd
   coordination). See `data/targets/active_sites/*.json`.
3. **Variant library (labels by construction).** For each target:

   *Positives (function preserved):*
   - WT.
   - Conservative point mutants outside the active-site mask.
   - **ProteinMPNN designs with the active-site mask fixed**, swept over
     temperatures (e.g. 0.1 / 0.3) and resampled per target. This is the
     dominant positive class.
   - Distant natural homologs preserving the catalytic motif (where
     available; optional for v1).

   *Negatives (function broken OR not in set):*
   - **Active-site knockouts** — mutate exactly the fixed-mask residues
     (ala-scan and charge-reversal). Fold usually preserved, function
     broken. Key hard-negative class for "function broken".
   - **Broad same-family homologs** — MMseqs2 hits against UniRef50 for
     each Ti, banded by seq-id (20-40% / 40-70%), deduplicated, and
     filtered to remove anything that clusters to another Ti. Key
     hard-negative class for "not in set": shares scaffold and often
     catalytic mechanism but is not one of our 9.
   - Composition-matched scrambles.
   - Heavy random perturbations (>30% edits).
   - Sequences drawn from unrelated folds (PDB decoys).
   - Random UniProt sample (easy negatives, coverage of deployed
     distribution).
   - High-temperature MPNN designs without the active-site mask
     (designed-to-fold, not-to-function).

4. **Oracle folding (QC only, not labeling).** Sample ~200-500 sequences
   per (target, class) cell and fold with ESMFold. Success criteria:
   - MPNN positives: TM to reference >= 0.8, pLDDT>=70 on active-site
     residues.
   - Active-site KOs: TM to reference >= 0.8 (fold preserved), distinct
     from positives by sequence change at the mask only.
   - Heavy perturbations: TM < 0.5 (fold lost).
   If QC fails for a cell, revisit generation for that cell before scaling.
5. **Student training.** Small dilated 1D CNN, ~0.5-2M params, 64-channel,
   5 dilated residual blocks, masked mean pool, 10-class softmax head
   (9 targets + "none"). At inference the binary decision is
   `1 - p(none)` vs a threshold; the multiclass argmax is the bonus ID.
   Single cross-entropy loss on the 10-way head. No positional
   embeddings. Balanced sampling across (target, class) cells. Optional
   auxiliary per-residue "is this a fixed-position residue" head as a
   training-time regulariser. Int8 quantize.
6. **Evaluation.**
   - **Primary** (binary in-set / not): ROC-AUC, PR-AUC, and operating
     point at a chosen threshold.
   - Primary hard cases: active-site KOs, broad same-family homologs
     (held out from training by UniRef50 cluster), active-site KOs of
     unseen MPNN designs.
   - **Secondary** (multiclass): per-target confusion matrix among
     predicted-positives. Inter-target confusion does not count against
     primary.
   - Baselines: BLAST/HMMER against the 9 targets (will false-positive
     on active-site KOs and on same-family homologs — this is the gap
     the student has to close), k-mer SVM, nearest-WT edit distance.
7. **Deployment.** Quantize, export to ONNX or TFLite, benchmark on Pi 4/5.
   Target: <10MB, <100ms/seq, <50MB RAM.

## Compute requirements

**Variant generation.** ProteinMPNN on a single GPU, ~seconds per design.
Negligible. A few hours for the full library across 9 targets.

**Oracle (ESMFold) folding — QC only.**

- ESMFold VRAM: ~14GB at 200 residues, scales ~quadratically. RTX 5000 Ada
  (32GB) handles most of our targets; T4/T5 (640-690 residues) need larger
  cards (H100, H200, B200).
- QC sample size: ~500/cell * 9 targets * ~5 classes ~= 22.5k folds. At
  3s/seq on an RTX 5000 Ada: ~20 GPU-hours total. Easily fits in one or
  two priority_gpu jobs.

**Student training.** Negligible. Any GPU or CPU for the smallest variants.
<1 GPU-hour end-to-end per training run.

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

- **MPNN designs are a design prior, not wet-lab truth.** Some "positive"
  variants will fold correctly and still be non-functional (allosteric
  disruption, altered dynamics, bad dynamics on conformational changes).
  Mitigation: state this clearly, treat labels as "function likely
  preserved" not "function measured". Cross-check with ESM-1v zero-shot
  fitness on a subset.
- **Active-site annotation gaps.** UniProt features are incomplete (T7 had
  none, T4 required manual extraction). Mitigation: PDB-ligand neighbor
  analysis, M-CSA fallback, exclude targets where even a manual pass can't
  produce a credible mask.
- **Sequence-similarity shortcut.** Student may just learn nearest-WT,
  missing active-site KOs entirely. Mitigation: active-site KOs are a
  first-class eval set; balanced sampling across (target, distance-bucket,
  class) cells.
- **Oracle mutation-insensitivity on QC.** ESMFold may call almost
  everything correctly folded, making "fold preserved?" QC toothless for
  KOs. Mitigation: active-site KO QC checks pLDDT on the mask residues,
  not just global TM.

## Milestones

1. Target set + active-site masks + oracle env smoke test. (done/in progress)
2. Single-target variant library (T2 or similar clean case): WT + MPNN
   positives + active-site KOs + scrambles. QC-fold a sample, check the
   three success criteria above.
3. Scale to all 9 targets with the full variant library.
4. Broad-negatives pipeline: MMseqs2 against UniRef50 per target,
   banded by seq-id, deduplicated, family-filtered. Same-family hard
   negatives are the demo-maker.
5. Student training end-to-end with baselines. Primary metric is binary
   AUC; secondary is multiclass per-target confusion.
6. Quantization, export, Pi benchmark.

## Repo layout

- `src/data/` variant generation, active-site mask lookup, dataset assembly
- `src/oracle/` ESMFold wrapper, TM-score computation (QC only)
- `src/student/` model defs, training, quantization, export
- `src/eval/` baselines, held-out evaluation, Pi benchmarks
- `scripts/` slurm submission, one-off jobs
- `configs/` experiment configs, target list, env specs
- `data/` git-tracked small inputs: reference FASTAs/PDBs,
  active-site masks
- `data/` gitignored scratch: variant libraries, oracle folds, checkpoints
- `notebooks/` exploration, QC, figures
