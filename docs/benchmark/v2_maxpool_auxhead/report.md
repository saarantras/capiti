# Benchmark report (operating point: FPR <= 0.05)

## Overall

| method | AUC | PR-AUC | acc | TPR | FPR | threshold |
|---|---|---|---|---|---|---|
| capiti_v2_maxpool_auxhead | 0.998 | 0.997 | 0.971 | 0.998 | 0.051 | 0.4854 |
| blast_nearest_wt | 0.702 | 0.551 | 0.523 | 0.000 | 0.051 | 805.8 |
| kmer3_nn | 0.617 | 0.468 | 0.523 | 0.000 | 0.051 | 0.4926 |
| kmer3_lr | 0.980 | 0.966 | 0.923 | 0.890 | 0.051 | 0.7819 |

## Binary accuracy per class

| class | n | capiti_v2_maxpool_auxhead | blast_nearest_wt | kmer3_nn | kmer3_lr |
|---|---|---|---|---|---|
| mpnn_positive | 900 | 0.998 | 0.000 | 0.000 | 0.890 |
| ala_scan | 82 | 0.573 | 0.988 | 0.890 | 0.317 |
| combined_ko | 161 | 1.000 | 1.000 | 0.994 | 1.000 |
| family_decoy | 165 | 1.000 | 1.000 | 1.000 | 1.000 |
| perturb30 | 270 | 0.922 | 0.796 | 0.830 | 1.000 |
| scramble | 270 | 1.000 | 1.000 | 1.000 | 1.000 |
| random_decoy | 157 | 1.000 | 1.000 | 1.000 | 1.000 |
