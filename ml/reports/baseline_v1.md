# Baseline model report: v1-29f9d2c5

TF-IDF (hashed word 1-2-grams) + logistic regression, isotonic calibration.
Data manifest sha256: `6df1667be3003fda9c23f83c30e20c9ccc3069ce053bf7f282dcb69ea91f9215`

## Metrics by split

The calibrator and thresholds are fit on the calib split, so its calibrated numbers are
in-sample. The test split is the honest estimate.

| | rows | unreliable share | macro-F1 | ROC-AUC | PR-AUC | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| val (raw) | 9195 | 0.4351 | 0.8749 | 0.9557 | 0.9456 | 0.085 | 0.0338 |
| val (calibrated) | 9195 | 0.4351 | 0.8755 | 0.9554 | 0.9409 | 0.0842 | 0.0163 |
| calib (raw) | 8530 | 0.4327 | 0.8922 | 0.9614 | 0.9531 | 0.0779 | 0.0362 |
| calib (calibrated) | 8530 | 0.4327 | 0.8933 | 0.9621 | 0.9498 | 0.0753 | 0.0025 |
| test (raw) | 8830 | 0.4341 | 0.873 | 0.9524 | 0.9429 | 0.0874 | 0.026 |
| test (calibrated) | 8830 | 0.4341 | 0.8737 | 0.9519 | 0.9379 | 0.0873 | 0.0166 |

## Test split by dataset (calibrated)

| | rows | unreliable share | macro-F1 | ROC-AUC | PR-AUC | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| fakenewsnet | 1524 | 0.3222 | 0.7181 | 0.8118 | 0.6813 | 0.1618 | 0.0531 |
| liar | 972 | 0.501 | 0.5857 | 0.624 | 0.6092 | 0.2658 | 0.1446 |
| welfake | 6334 | 0.4507 | 0.9454 | 0.9866 | 0.9813 | 0.042 | 0.0192 |

## Test split by length (calibrated)

| | rows | unreliable share | macro-F1 | ROC-AUC | PR-AUC | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| article | 5291 | 0.4544 | 0.9455 | 0.9863 | 0.981 | 0.0429 | 0.0222 |
| headline | 2742 | 0.4362 | 0.7027 | 0.7901 | 0.7412 | 0.1918 | 0.0822 |
| short | 797 | 0.2923 | 0.9638 | 0.9956 | 0.9887 | 0.0229 | 0.0261 |

## Verdict thresholds

`t_low = 0.35`, `t_high = 0.65` (band targets met on calib: True).

| band on test | share of rows | unreliable rate |
|---|---:|---:|
| reliable | 0.5299 | 0.0898 |
| potentially_misleading | 0.1042 | 0.513 |
| high_risk | 0.3659 | 0.9102 |

## Strongest features

**Toward unreliable:** `via`, `read more`, `video`, `the us`, `october`, `2016`, `<bucket 1043942>`, `breaking`, `<bucket 396043>`, `november 2016`, `november`, `hillary`, `read`, `via breitbart`, `on october`, `wow`, `un`, `react`, `swipe left`, `us`, `на`, `<bucket 160149>`, `sputnik`, `caitlyn`, `<bucket 633143>`, `anti`, `<bucket 282532>`, `26 2016`, `27 2016`, `rt`, `<bucket 599892>`, `debunked`, `source`, `for entire`, `rumor`, `obama`, `2016 by`, `<bucket 704098>`, `below`, `<bucket 879778>`, `что`, `illegals`, `begging`, `11 16`, `<bucket 535980>`, `<bucket 721668>`, `<bucket 125041>`, `cops`, `mosul`, `wisconsin`

**Toward reliable:** `breitbart`, `president donald`, `lobster`, `said on`, `york times`, `follow`, `pic twitter`, `https`, `music video`, `pic`, `on twitter`, `<bucket 374404>`, `season`, `<bucket 1002616>`, `on friday`, `<bucket 627274>`, `said`, `on tuesday`, `on thursday`, `on monday`, `follow him`, `trailer`, `on wednesday`, `<bucket 717291>`, `mr`, `the arrangement`, `the new`, `said in`, `twitter`, `ms`, `for breitbart`, `<bucket 296882>`, `breitbart the`, `he said`, `or email`, `<bucket 1029517>`, `<bucket 132919>`, `netflix`, `in 2016`, `<bucket 832669>`, `beauty`, `thursday`, `shirtless`, `breitbart texas`, `pam key`, `follow pam`, `<bucket 658266>`, `key on`, `friday`, `2018`
