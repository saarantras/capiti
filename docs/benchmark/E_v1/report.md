# Benchmark report (operating point: FPR <= 0.05)

## Overall

| method | AUC | PR-AUC | acc | TPR | FPR | threshold |
|---|---|---|---|---|---|---|
| capiti_E_v1 | 0.997 | 0.997 | 0.980 | 1.000 | 0.050 | 0.1831 |
| capiti_E_v1+gate | 0.984 | 0.993 | 0.970 | 0.983 | 0.050 | 0.005329 |
| blast_nearest_wt | 0.581 | 0.598 | 0.377 | 0.000 | 0.052 | 740 |
| kmer3_nn | 0.410 | 0.512 | 0.377 | 0.000 | 0.050 | 0.4457 |
| kmer3_lr | 0.975 | 0.979 | 0.883 | 0.839 | 0.050 | 0.8106 |

## Binary accuracy per class

| class | n | capiti_E_v1 | capiti_E_v1+gate | blast_nearest_wt | kmer3_nn | kmer3_lr |
|---|---|---|---|---|---|---|
| mpnn_positive | 5900 | 1.000 | 0.983 | 0.000 | 0.000 | 0.839 |
| ala_scan | 466 | 0.637 | 0.966 | 0.974 | 0.961 | 0.597 |
| combined_ko | 900 | 0.999 | 0.987 | 0.999 | 0.993 | 0.993 |
| perturb30 | 1170 | 0.979 | 0.868 | 0.839 | 0.854 | 1.000 |
| scramble | 1170 | 1.000 | 0.990 | 1.000 | 1.000 | 1.000 |
| random_decoy | 181 | 1.000 | 0.994 | 1.000 | 1.000 | 0.994 |
