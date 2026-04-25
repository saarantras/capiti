# Benchmark report (operating point: FPR <= 0.05)

## Overall

| method | AUC | PR-AUC | acc | TPR | FPR | threshold |
|---|---|---|---|---|---|---|
| capiti_C_v1 | 0.986 | 0.986 | 0.957 | 0.963 | 0.050 | 0.9944 |
| capiti_C_v1+gate | 0.970 | 0.984 | 0.958 | 0.964 | 0.050 | 0.02341 |
| blast_nearest_wt | 0.572 | 0.542 | 0.431 | 0.015 | 0.050 | 521 |
| kmer3_nn | 0.477 | 0.491 | 0.423 | 0.000 | 0.050 | 0.451 |
| kmer3_lr | 0.974 | 0.972 | 0.878 | 0.820 | 0.050 | 0.8215 |

## Binary accuracy per class

| class | n | capiti_C_v1 | capiti_C_v1+gate | blast_nearest_wt | kmer3_nn | kmer3_lr |
|---|---|---|---|---|---|---|
| mpnn_positive | 22399 | 0.963 | 0.964 | 0.015 | 0.000 | 0.820 |
| ala_scan | 1540 | 0.460 | 0.953 | 0.947 | 0.943 | 0.537 |
| combined_ko | 2968 | 0.980 | 0.945 | 0.984 | 0.991 | 0.939 |
| perturb30 | 6574 | 0.999 | 0.943 | 0.883 | 0.881 | 1.000 |
| scramble | 6690 | 1.000 | 0.958 | 1.000 | 1.000 | 1.000 |
| random_decoy | 181 | 1.000 | 0.967 | 1.000 | 1.000 | 0.994 |
