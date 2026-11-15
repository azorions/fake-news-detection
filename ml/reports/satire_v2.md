# Satire headline scorer: v2-89b967c5

TF-IDF (hashed word 1-2-grams of the first 64 words) + logistic regression, isotonic calibration, trained on headlines only.
Data manifest sha256: `b02837724d857df5685684c3fff4aaa22fef7a3d03be4a9bf26163f7efe10a78`

## What the score means

Every satire headline in the data comes from one publisher and every other headline from another, so the score partly reflects the house style of these two publishers, not satire as such. It is a hint shown next to the verdict, not a verdict.

| publisher | rows | satire share |
|---|---:|---:|
| huffingtonpost.com | 14914 | 0.0 |
| theonion.com | 13592 | 1.0 |

## Metrics

The rows are split 80/10/10 by text hash: 22778 train, 2859 calib, 2869 test. The calibrator is fit on the calib tenth, so its calibrated numbers are in-sample. Nothing is fit on the test tenth (`eval/satire_slice.parquet`).

| | rows | satire share | macro-F1 | ROC-AUC | PR-AUC | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| calib (raw) | 2859 | 0.482 | 0.8471 | 0.9255 | 0.9191 | 0.1141 | 0.0711 |
| calib (calibrated) | 2859 | 0.482 | 0.8508 | 0.928 | 0.9137 | 0.1058 | 0.0011 |
| test (raw) | 2869 | 0.4779 | 0.8493 | 0.9229 | 0.9133 | 0.1148 | 0.0606 |
| test (calibrated) | 2869 | 0.4779 | 0.8425 | 0.92 | 0.9012 | 0.1118 | 0.0271 |

## Strongest features

**Toward satire:** `nation`, `area`, `report`, `man`, `local`, `fucking`, `onion`, `announces`, `fuck`, `introduces`, `clearly`, `guy`, `wondering`, `study`, `only`, `shit`, `the onion`, `few`, `god`, `unveils`, `audience`, `enough`, `coworker`, `kavanaugh`, `bar`

**Toward not satire:** `the`, `is`, `and`, `an`, `trump`, `why`, `my`, `your`, `watch`, `are`, `california`, `how`, `our`, `says he`, `donald`, `these`, `how to`, `cops`, `accused`, `this`, `tips`, `huffpost`, `reportedly`, `china`, `lessons`
