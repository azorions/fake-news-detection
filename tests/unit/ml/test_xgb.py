from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score

from fnd_ml.common.artifacts import resolve
from fnd_ml.common.features import FEATURE_NAMES
from fnd_ml.models import xgb
from fnd_ml.models.xgb import XgbConfig, XgbModel, train_xgb
from tests.fixtures.processed import stylistic_corpus, two_dataset_corpus, write_processed

SMALL = XgbConfig(n_features=2**12, svd_components=8, n_estimators=60)


def test_xgb_learns_a_stylistic_rule() -> None:
    frame = stylistic_corpus(400, seed=3)
    train, val, calib, test = (
        frame.iloc[:200],
        frame.iloc[200:260],
        frame.iloc[260:330],
        frame.iloc[330:],
    )

    scorer = xgb.fit_scorer(train, val, calib, config=SMALL)

    predicted = (scorer(test) >= 0.5).astype(int)
    assert f1_score(test["label"], predicted, average="macro") > 0.9


@pytest.fixture(scope="module")
def fitted() -> tuple[XgbModel, list[str]]:
    frame = two_dataset_corpus(300, seed=5)
    train, val, held_out = frame.iloc[:180], frame.iloc[180:240], frame.iloc[240:]
    model = XgbModel(SMALL).fit(
        train["model_text"].tolist(), train["label"], val["model_text"].tolist(), val["label"]
    )
    texts = held_out["model_text"].tolist()
    model.calibrate_scores(model.predict_raw(texts), held_out["label"])
    return model, texts


def test_xgb_save_and_load_give_identical_probabilities(
    fitted: tuple[XgbModel, list[str]], tmp_path: Path
) -> None:
    model, texts = fitted

    model.save(tmp_path / "artifact", {"version": "test-1"})
    loaded = XgbModel.load(tmp_path / "artifact")

    np.testing.assert_allclose(loaded.predict_proba(texts), model.predict_proba(texts), atol=1e-6)
    assert loaded.metadata["version"] == "test-1"
    assert sorted(p.name for p in (tmp_path / "artifact").iterdir()) == [
        "booster.json",
        "features.npz",
        "metadata.json",
    ]


def test_xgb_probabilities_are_calibrated_into_the_allowed_range(
    fitted: tuple[XgbModel, list[str]],
) -> None:
    model, texts = fitted
    calibrated = model.predict_proba(texts)
    assert calibrated.min() >= 0.01
    assert calibrated.max() <= 0.99
    order = np.argsort(model.predict_raw(texts))
    assert (np.diff(calibrated[order]) >= 0).all()


def test_an_unfitted_xgb_model_refuses_to_predict() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        XgbModel(SMALL).predict_raw(["some text here"])


def test_training_xgb_writes_an_artifact_a_registry_entry_and_a_report(tmp_path: Path) -> None:
    processed, lock = write_processed(tmp_path, two_dataset_corpus())

    report = train_xgb(
        processed, tmp_path / "artifacts", tmp_path / "reports", SMALL, lock_path=lock
    )

    artifact = resolve(tmp_path / "artifacts", "xgb")
    assert artifact is not None
    assert {p.name for p in artifact.iterdir()} == {
        "booster.json",
        "features.npz",
        "metadata.json",
        "thresholds.json",
    }
    assert resolve(tmp_path / "artifacts", "baseline") is None
    assert report["model"] == "xgb"
    assert report["version"] == artifact.name
    assert report["version"].startswith("v2-")
    assert report["metrics"]["test"]["calibrated"]["macro_f1"] > 0.9
    assert set(report["metrics"]["test_by_dataset"]) == {"alpha", "beta"}
    assert set(report["metrics"]["test_by_length"]) == {"headline"}
    assert report["rows"] == {"train": 240, "val": 50, "calib": 60, "test": 50}
    known = {*FEATURE_NAMES, *(f"svd_{i}" for i in range(SMALL.svd_components))}
    assert {row["feature"] for row in report["feature_importance"]} <= known
    assert report["feature_importance"][0]["gain"] > 0
    assert 1 <= report["trees"] <= SMALL.n_estimators
    markdown = (tmp_path / "reports" / "xgb_v2.md").read_text(encoding="utf-8")
    assert "macro-F1" in markdown
    assert "Test split by dataset" in markdown
    assert "## Feature importance" in markdown
