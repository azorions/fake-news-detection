# Baseline model report: v1-69c6a3df

TF-IDF (hashed word 1-2-grams) + logistic regression, isotonic calibration.
Data manifest sha256: `ec22b4f5ed731fcb17263cfb9fb37d262142577ad47363ea60abf65fa6a90a34`

## Metrics by split

The calibrator and thresholds are fit on the calib split, so its calibrated numbers are
in-sample. The test split is the honest estimate.

| | rows | unreliable share | macro-F1 | ROC-AUC | PR-AUC | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| val (raw) | 9121 | 0.4214 | 0.86 | 0.9431 | 0.9265 | 0.0953 | 0.0268 |
| val (calibrated) | 9121 | 0.4214 | 0.8568 | 0.9425 | 0.9191 | 0.0949 | 0.0145 |
| calib (raw) | 8429 | 0.4379 | 0.8698 | 0.9498 | 0.9405 | 0.0896 | 0.033 |
| calib (calibrated) | 8429 | 0.4379 | 0.8724 | 0.9506 | 0.9364 | 0.0875 | 0.002 |
| test (raw) | 8794 | 0.4289 | 0.8546 | 0.9415 | 0.928 | 0.0973 | 0.0204 |
| test (calibrated) | 8794 | 0.4289 | 0.8546 | 0.941 | 0.9214 | 0.0975 | 0.0159 |

## Test split by dataset (calibrated)

| | rows | unreliable share | macro-F1 | ROC-AUC | PR-AUC | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| fakenewsnet | 1521 | 0.3228 | 0.7225 | 0.8135 | 0.6854 | 0.1587 | 0.0393 |
| liar | 972 | 0.501 | 0.575 | 0.6253 | 0.6139 | 0.2597 | 0.1435 |
| welfake | 6301 | 0.4434 | 0.9211 | 0.9778 | 0.968 | 0.0577 | 0.0173 |

## Test split by length (calibrated)

| | rows | unreliable share | macro-F1 | ROC-AUC | PR-AUC | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| article | 5262 | 0.4447 | 0.9193 | 0.9771 | 0.9673 | 0.0591 | 0.0231 |
| headline | 2737 | 0.4366 | 0.6985 | 0.7888 | 0.7404 | 0.1895 | 0.0684 |
| short | 795 | 0.2981 | 0.946 | 0.9918 | 0.9797 | 0.035 | 0.0358 |

## Verdict thresholds

`t_low = 0.35`, `t_high = 0.65` (band targets met on calib: True).

| band on test | share of rows | unreliable rate |
|---|---:|---:|
| reliable | 0.518 | 0.0929 |
| potentially_misleading | 0.1216 | 0.4883 |
| high_risk | 0.3605 | 0.8918 |

## Strongest features

**Toward unreliable:** `video`, `the us`, `breaking`, `<bucket 396043>`, `<bucket 1043942>`, `hillary`, `<bucket 160149>`, `<bucket 879778>`, `wow`, `anti`, `un`, `below`, `<bucket 282532>`, `us`, `caitlyn`, `rumor`, `<bucket 633143>`, `cops`, `via`, `<bucket 535980>`, `illegals`, `debunked`, `obama`, `mosul`, `<bucket 125041>`, `react`, `kanye`, `wisconsin`, `begging`, `<bucket 39514>`, `<bucket 574323>`, `video the`, `<bucket 599892>`, `<bucket 498633>`, `massive`, `<bucket 864930>`, `download`, `october`, `lol`, `awesome`, `however`, `<bucket 852768>`, `<bucket 190390>`, `says president`, `brad`, `aleppo`, `share on`, `doj`, `fbi`, `non`

**Toward reliable:** `president donald`, `lobster`, `said on`, `milo`, `music video`, `mr`, `<bucket 1002616>`, `<bucket 374404>`, `on friday`, `said`, `ms`, `season`, `on thursday`, `<bucket 627274>`, `on tuesday`, `awards`, `on monday`, `trailer`, `friday`, `on wednesday`, `said in`, `thursday`, `<bucket 717291>`, `the arrangement`, `<bucket 296882>`, `beauty`, `<bucket 1029517>`, `finale`, `shirtless`, `netflix`, `he said`, `easter`, `<bucket 658266>`, `on sunday`, `<bucket 615575>`, `<bucket 115688>`, `saturday`, `tribute to`, `islamic state`, `pink`, `<bucket 128009>`, `<bucket 991971>`, `tattoo`, `<bucket 862180>`, `<bucket 168100>`, `mtv`, `tuesday`, `<bucket 746470>`, `<bucket 832669>`, `<bucket 447376>`
