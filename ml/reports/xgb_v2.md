# XGBoost model report: v2-c5006e2d

XGBoost over 12 stylometric features and a 256-component SVD of the hashed TF-IDF, isotonic calibration.
Data manifest sha256: `849a3aa99343da605d6db7415594cfb545d8bfa39a51b034ea810d07e686f539`

## Metrics by split

The calibrator and thresholds are fit on the calib split, so its calibrated numbers are
in-sample. The test split is the honest estimate.

| | rows | unreliable share | macro-F1 | ROC-AUC | PR-AUC | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| val (raw) | 9058 | 0.4181 | 0.8765 | 0.9606 | 0.952 | 0.0803 | 0.0286 |
| val (calibrated) | 9058 | 0.4181 | 0.8809 | 0.9601 | 0.9473 | 0.0785 | 0.0136 |
| calib (raw) | 8359 | 0.4323 | 0.8829 | 0.9637 | 0.9578 | 0.077 | 0.024 |
| calib (calibrated) | 8359 | 0.4323 | 0.8889 | 0.9641 | 0.9543 | 0.0746 | 0.004 |
| test (raw) | 8704 | 0.4236 | 0.8707 | 0.9559 | 0.948 | 0.0846 | 0.0276 |
| test (calibrated) | 8704 | 0.4236 | 0.8743 | 0.9553 | 0.9426 | 0.083 | 0.011 |

## Test split by dataset (calibrated)

| | rows | unreliable share | macro-F1 | ROC-AUC | PR-AUC | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| fakenewsnet | 1521 | 0.3228 | 0.6845 | 0.7775 | 0.6492 | 0.1687 | 0.0397 |
| liar | 972 | 0.501 | 0.5947 | 0.613 | 0.6101 | 0.2532 | 0.0818 |
| welfake | 6211 | 0.4362 | 0.9518 | 0.9905 | 0.9869 | 0.0354 | 0.0154 |

## Test split by length (calibrated)

| | rows | unreliable share | macro-F1 | ROC-AUC | PR-AUC | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| article | 5178 | 0.4365 | 0.9541 | 0.9914 | 0.9881 | 0.0338 | 0.02 |
| headline | 2736 | 0.4364 | 0.6962 | 0.7735 | 0.7358 | 0.1912 | 0.0452 |
| short | 790 | 0.2949 | 0.9486 | 0.9886 | 0.9784 | 0.0312 | 0.0243 |

## Verdict thresholds

`t_low = 0.35`, `t_high = 0.65` (band targets met on calib: True).

| band on test | share of rows | unreliable rate |
|---|---:|---:|
| reliable | 0.5315 | 0.085 |
| potentially_misleading | 0.1347 | 0.4744 |
| high_risk | 0.3339 | 0.9422 |

## Feature importance

545 trees after early stopping on val. Mean gain of the splits that use a feature; `svd_N` is the N-th SVD component of the TF-IDF.

| feature | gain |
|---|---:|
| `std_sentence_words` | 196.2441 |
| `svd_1` | 104.4946 |
| `second_person_share` | 93.1642 |
| `caps_share` | 63.6472 |
| `question_per_100w` | 61.689 |
| `exclaim_per_100w` | 53.5088 |
| `quote_per_100w` | 49.8461 |
| `mean_sentence_words` | 39.721 |
| `svd_13` | 36.4028 |
| `log_words` | 34.5325 |
| `svd_10` | 33.2165 |
| `svd_18` | 27.8933 |
| `svd_2` | 23.6675 |
| `svd_4` | 20.3844 |
| `svd_7` | 20.2296 |
| `svd_48` | 20.1461 |
| `title_case_share` | 18.4827 |
| `svd_12` | 18.2155 |
| `svd_0` | 16.8162 |
| `svd_33` | 16.4831 |
