# Benchmark report (operating point: FPR <= 0.05)

## Overall

| method | AUC | PR-AUC | acc | TPR | FPR | threshold |
|---|---|---|---|---|---|---|
| capiti_v3 | 0.997 | 0.996 | 0.972 | 1.000 | 0.050 | 0.05639 |
| capiti_v3+gate | 1.000 | 1.000 | 0.972 | 1.000 | 0.050 | 0.004661 |
| blast_nearest_wt | 0.705 | 0.550 | 0.524 | 0.000 | 0.050 | 805.2 |
| kmer3_nn | 0.620 | 0.469 | 0.524 | 0.000 | 0.050 | 0.4696 |
| kmer3_lr | 0.979 | 0.963 | 0.910 | 0.862 | 0.050 | 0.7705 |

## Binary accuracy per class

| class | n | capiti_v3 | capiti_v3+gate | blast_nearest_wt | kmer3_nn | kmer3_lr |
|---|---|---|---|---|---|---|
| mpnn_positive | 900 | 1.000 | 1.000 | 0.000 | 0.000 | 0.862 |
| ala_scan | 86 | 0.465 | 1.000 | 0.965 | 0.895 | 0.349 |
| combined_ko | 161 | 1.000 | 1.000 | 0.994 | 0.981 | 1.000 |
| family_decoy | 165 | 1.000 | 1.000 | 1.000 | 0.994 | 1.000 |
| perturb30 | 270 | 0.963 | 0.800 | 0.807 | 0.841 | 1.000 |
| scramble | 270 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| random_decoy | 157 | 1.000 | 0.987 | 1.000 | 1.000 | 1.000 |
