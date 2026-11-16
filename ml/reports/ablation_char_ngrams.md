# Ablation: character n-grams in the baseline

Both variants are fit on the same stratified sample of 20000 train rows (of 64681) and scored on the whole val split (9058 rows), uncalibrated.
Data manifest sha256: `849a3aa99343da605d6db7415594cfb545d8bfa39a51b034ea810d07e686f539`

Peak memory is the growth of the process's resident memory while fitting and scoring. The projection for the whole train split scales it by the row ratio.

| features | val macro-F1 | val ROC-AUC | peak memory (MB) | projected, full train (GB) |
|---|---:|---:|---:|---:|
| word 1-2 | 0.8383 | 0.9271 | 345.6 | 1.09 |
| word 1-2 + char_wb 3-5 | 0.835 | 0.9212 | 999.9 | 3.16 |

## Decision

Character n-grams are **not adopted**: val macro-F1 changes by -0.0033. The rule, fixed before the run, is to adopt them only for a gain of at least 0.01 with projected memory under 6 GB.
