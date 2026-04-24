# Benchmark report (operating point: FPR <= 0.05)

## Overall

| method | AUC | PR-AUC | acc | TPR | FPR | threshold |
|---|---|---|---|---|---|---|
| capiti | 0.985 | 0.972 | 0.968 | 0.990 | 0.051 | 0.8128 |
| blast_nearest_wt | 0.697 | 0.543 | 0.524 | 0.000 | 0.051 | 818.9 |
| kmer3_nn | 0.622 | 0.470 | 0.524 | 0.000 | 0.051 | 0.4922 |
| kmer3_lr | 0.981 | 0.963 | 0.942 | 0.933 | 0.051 | 0.7371 |

## Binary accuracy per class

| class | n | capiti | blast_nearest_wt | kmer3_nn | kmer3_lr |
|---|---|---|---|---|---|
| mpnn_positive | 900 | 0.990 | 0.000 | 0.000 | 0.933 |
| ala_scan | 85 | 0.365 | 0.953 | 0.906 | 0.341 |
| combined_ko | 161 | 0.994 | 0.994 | 0.988 | 1.000 |
| family_decoy | 165 | 1.000 | 1.000 | 1.000 | 1.000 |
| perturb30 | 270 | 0.996 | 0.811 | 0.830 | 1.000 |
| scramble | 270 | 1.000 | 1.000 | 1.000 | 1.000 |
| random_decoy | 157 | 1.000 | 1.000 | 1.000 | 1.000 |
