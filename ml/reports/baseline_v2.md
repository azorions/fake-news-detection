# Baseline model report: v2-7fda82b8

TF-IDF (hashed word 1-2-grams) + logistic regression, isotonic calibration.
Data manifest sha256: `849a3aa99343da605d6db7415594cfb545d8bfa39a51b034ea810d07e686f539`

## Metrics by split

The calibrator and thresholds are fit on the calib split, so its calibrated numbers are
in-sample. The test split is the honest estimate.

| | rows | unreliable share | macro-F1 | ROC-AUC | PR-AUC | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| val (raw) | 9058 | 0.4181 | 0.8579 | 0.9424 | 0.9247 | 0.0958 | 0.0267 |
| val (calibrated) | 9058 | 0.4181 | 0.8567 | 0.9417 | 0.9171 | 0.0956 | 0.0141 |
| calib (raw) | 8359 | 0.4323 | 0.8681 | 0.9486 | 0.9376 | 0.0905 | 0.0308 |
| calib (calibrated) | 8359 | 0.4323 | 0.8709 | 0.9495 | 0.9332 | 0.0885 | 0.002 |
| test (raw) | 8704 | 0.4236 | 0.853 | 0.9405 | 0.9253 | 0.0979 | 0.0204 |
| test (calibrated) | 8704 | 0.4236 | 0.8535 | 0.9402 | 0.9184 | 0.098 | 0.014 |

## Test split by dataset (calibrated)

| | rows | unreliable share | macro-F1 | ROC-AUC | PR-AUC | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| fakenewsnet | 1521 | 0.3228 | 0.7182 | 0.8129 | 0.6835 | 0.1593 | 0.0419 |
| liar | 972 | 0.501 | 0.5807 | 0.6296 | 0.6152 | 0.2579 | 0.1368 |
| welfake | 6211 | 0.4362 | 0.9209 | 0.9773 | 0.9665 | 0.058 | 0.0168 |

## Test split by length (calibrated)

| | rows | unreliable share | macro-F1 | ROC-AUC | PR-AUC | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| article | 5178 | 0.4365 | 0.9189 | 0.9765 | 0.9654 | 0.0595 | 0.0216 |
| headline | 2736 | 0.4364 | 0.6995 | 0.7897 | 0.7398 | 0.1892 | 0.0662 |
| short | 790 | 0.2949 | 0.9452 | 0.9919 | 0.9794 | 0.0349 | 0.0342 |

## Verdict thresholds

`t_low = 0.35`, `t_high = 0.65` (band targets met on calib: True).

| band on test | share of rows | unreliable rate |
|---|---:|---:|
| reliable | 0.5218 | 0.092 |
| potentially_misleading | 0.1228 | 0.4892 |
| high_risk | 0.3554 | 0.8878 |

## Leak probes

Label probe: none of the 100 strongest features names a source.

Origin probe: a linear model names the dataset of a scrubbed val text with accuracy 0.9663 (always answering the largest dataset gives 0.6755); within label 0: 0.978, within label 1: 0.9482. Reported, not gated.

## Strongest features

**Toward unreliable:** `the us`, `video`, `breaking`, `celebrities react`, `despite report`, `hillary`, `le gorafi`, `gorafi`, `wow`, `un`, `anti`, `below`, `us`, `pattinson`, `via`, `rumor`, `kristen stewart`, `cops`, `caitlyn`, `illegals`, `mosul`, `obama`, `advisers have`, `debunked`, `react`, `tom cruise`, `wisconsin`, `begging`, `kanye`, `video the`, `massive`, `katie holmes`, `bf`, `chart of`, `snubbed`, `rumours`, `october`, `download`, `says president`, `however`, `lol`, `awesome`, `aleppo`, `weisz`, `doj`, `fbi`, `non`, `beyonce`, `brad`, `rihanna`

**Toward reliable:** `president donald`, `is us`, `said on`, `music video`, `met gala`, `cardi`, `on friday`, `mr`, `said`, `ms`, `season`, `baby no`, `on thursday`, `on tuesday`, `on monday`, `awards`, `trailer`, `friday`, `on wednesday`, `bachelorette`, `said in`, `thursday`, `the arrangement`, `beauty`, `finale`, `wags`, `riverdale`, `shirtless`, `netflix`, `makeup`, `easter`, `he said`, `on sunday`, `hollywood medium`, `bachelor`, `saturday`, `pics`, `tribute to`, `pink`, `islamic state`, `teen mom`, `revenge body`, `tattoo`, `tuesday`, `mtv`, `says donald`, `tour`, `transcript as`, `recap`, `teases`
