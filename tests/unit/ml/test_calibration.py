import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from fnd_ml.calibration import run as calibration_run
from fnd_ml.calibration.metrics import brier, expected_calibration_error
from fnd_ml.calibration.run import calibrate_run
from fnd_ml.calibration.temperature import apply_temperature, fit_temperature, sigmoid
from fnd_ml.calibration.thresholds import pick_thresholds
from fnd_ml.cli import app
from fnd_ml.common import resources
from fnd_ml.data.build import MANIFEST_NAME, ManifestMismatch
from fnd_ml.data.download import sha256_file
from fnd_ml.eval.metrics import classification_report
from tests.fixtures.processed import write_processed
from tests.fixtures.synthetic import make_corpus


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


def _logits_and_labels(scale: float, n: int = 5000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Labels drawn from ``sigmoid(z)``; the model reports ``scale * z`` as its logit."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 2.0, n)
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-z))).astype(int)
    return scale * z, y


def test_overconfident_logits_get_their_temperature_back() -> None:
    logits, y = _logits_and_labels(3.0)

    temperature = fit_temperature(logits, y)

    assert temperature == pytest.approx(3.0, rel=0.10)
    before = expected_calibration_error(y, apply_temperature(logits, 1.0))
    after = expected_calibration_error(y, apply_temperature(logits, temperature))
    assert after < before


def test_calibrated_logits_keep_a_temperature_of_one() -> None:
    logits, y = _logits_and_labels(1.0)

    assert fit_temperature(logits, y) == pytest.approx(1.0, rel=0.10)


def test_the_temperature_stays_inside_its_bounds() -> None:
    separable = np.array([-1.0] * 50 + [1.0] * 50)
    labels = np.array([0] * 50 + [1] * 50)

    # Perfectly separated logits want T -> 0; labels unrelated to the logits want T -> infinity.
    assert fit_temperature(separable, labels) == pytest.approx(0.05, rel=0.01)
    assert fit_temperature(separable, np.tile([0, 1], 50)) == pytest.approx(20.0, rel=0.01)


def test_apply_temperature_never_leaves_the_clipped_range() -> None:
    p = apply_temperature(np.array([-1e6, -50.0, 0.0, 50.0, 1e6]), 0.05)

    assert p.min() == pytest.approx(0.01)
    assert p.max() == pytest.approx(0.99)
    assert p[2] == pytest.approx(0.5)
    assert np.all(np.diff(p) >= 0)


def test_sigmoid_is_not_clipped_and_does_not_overflow() -> None:
    with np.errstate(over="raise"):
        p = sigmoid(np.array([-1e6, 0.0, 1e6]))

    np.testing.assert_allclose(p, [0.0, 0.5, 1.0])


# ---------------------------------------------------------------- fnd calibrate


def _finished_run(root: Path, manifest_sha: str) -> Path:
    run_dir = root / "run"
    (run_dir / "best").mkdir(parents=True)
    summary = {"data_manifest_sha256": manifest_sha}
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return run_dir


def _overconfident(model_dir: Path, frame: pd.DataFrame, **_: object) -> np.ndarray:
    """Stands in for the model: right about four rows in five, and far too sure every time."""
    rng = np.random.default_rng(0)
    side = np.where(rng.uniform(size=len(frame)) < 0.8, 1.0, -1.0)
    return 8.0 * side * (2.0 * frame["label"].to_numpy() - 1.0)


def test_calibrate_run_uses_the_calib_split_and_writes_both_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    processed, lock = write_processed(tmp_path, make_corpus(400))
    run_dir = _finished_run(tmp_path, sha256_file(processed / MANIFEST_NAME))
    seen: dict[str, object] = {}

    def logits(model_dir: Path, frame: pd.DataFrame, **options: object) -> np.ndarray:
        seen.update(model_dir=model_dir, ids=frame["text"].tolist())
        return _overconfident(model_dir, frame)

    monkeypatch.setattr(calibration_run, "predict_logits", logits)

    result = calibrate_run(run_dir, processed, lock_path=lock)

    calib = pd.read_parquet(processed / "splits" / "calib.parquet")
    assert seen == {"model_dir": run_dir / "best", "ids": calib["text"].tolist()}
    calibration = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
    assert calibration == result["calibration"]
    assert calibration["method"] == "temperature"
    assert calibration["rows"] == len(calib)
    assert 1.5 < calibration["temperature"] <= 20.0
    assert calibration["ece_after"] < calibration["ece_before"]
    assert calibration["brier_after"] < calibration["brier_before"]
    thresholds = json.loads((run_dir / "thresholds.json").read_text(encoding="utf-8"))
    assert thresholds == result["thresholds"]
    assert thresholds["t_low"] <= 0.35
    assert thresholds["t_high"] >= 0.65
    assert np.load(run_dir / "calib_logits.npy").shape == (len(calib),)


def test_calibrate_run_refuses_a_model_trained_on_another_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    processed, lock = write_processed(tmp_path, make_corpus(400))
    run_dir = _finished_run(tmp_path, "0" * 64)
    monkeypatch.setattr(calibration_run, "predict_logits", _overconfident)

    with pytest.raises(ManifestMismatch, match="was trained on data manifest 000000000000"):
        calibrate_run(run_dir, processed, lock_path=lock)

    assert not (run_dir / "calibration.json").exists()


def test_the_calibrate_command_prints_the_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(resources, "free_ram_gb", lambda: 64.0)
    captured: dict[str, Path] = {}

    def fake(run_dir: Path, processed_dir: Path) -> dict[str, object]:
        captured.update(run_dir=run_dir, processed_dir=processed_dir)
        return {"calibration": {"temperature": 1.7}, "thresholds": {"t_low": 0.3, "t_high": 0.7}}

    monkeypatch.setattr(calibration_run, "calibrate_run", fake)

    result = CliRunner().invoke(app, ["calibrate", "--run", str(tmp_path / "run")])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["calibration"]["temperature"] == 1.7
    assert captured == {"run_dir": tmp_path / "run", "processed_dir": Path("data/processed")}
