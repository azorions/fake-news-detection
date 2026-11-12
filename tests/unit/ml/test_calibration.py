import numpy as np
import pytest

from fnd_ml.calibration.metrics import brier, expected_calibration_error
from fnd_ml.calibration.thresholds import pick_thresholds
from fnd_ml.eval.metrics import classification_report


def test_ece_is_near_zero_for_calibrated_probabilities() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20_000)
    y = (rng.uniform(0, 1, 20_000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.02


def test_ece_is_large_for_confident_wrong_probabilities() -> None:
    assert expected_calibration_error(np.zeros(100, dtype=int), np.full(100, 0.9)) > 0.8


def test_brier_is_the_mean_squared_error_of_the_probability() -> None:
    assert brier(np.array([1, 0]), np.array([0.8, 0.3])) == pytest.approx(0.065)


def test_thresholds_keep_a_middle_band_even_when_the_model_separates_perfectly() -> None:
    y = np.array([0] * 500 + [1] * 500)
    p = np.concatenate([np.full(500, 0.02), np.full(500, 0.98)])

    result = pick_thresholds(y, p)

    assert (result.t_low, result.t_high) == (0.35, 0.65)
    assert result.met_targets


def test_thresholds_tighten_until_the_band_precision_target_is_met() -> None:
    # 300 rows at p=0.70 of which only half are unreliable, 300 rows at p=0.95 that all are.
    y = np.array([0] * 400 + [0] * 150 + [1] * 150 + [1] * 300)
    p = np.concatenate([np.full(400, 0.05), np.full(300, 0.70), np.full(300, 0.95)])

    result = pick_thresholds(y, p, precision_target=0.90)

    assert 0.70 < result.t_high <= 0.95
    assert y[p >= result.t_high].mean() >= 0.90
    assert result.met_targets


def test_thresholds_fall_back_and_say_so_when_no_band_reaches_the_target() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2_000)
    p = rng.uniform(0, 1, 2_000)  # unrelated to the labels

    result = pick_thresholds(y, p)

    assert (result.t_low, result.t_high) == (0.35, 0.65)
    assert not result.met_targets


def test_classification_report_has_the_headline_metrics() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])

    report = classification_report(y, p)

    assert report["n"] == 4
    assert report["macro_f1"] == pytest.approx(1.0)
    assert report["roc_auc"] == pytest.approx(1.0)
    assert set(report) >= {"pr_auc", "brier", "ece", "unreliable_share"}
