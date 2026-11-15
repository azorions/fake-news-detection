from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fnd_ml.data.build import MANIFEST_NAME, ManifestMismatch
from fnd_ml.models import baseline, xgb
from fnd_ml.models.baseline import BaselineConfig, BaselineModel
from fnd_ml.models.scorer import Fit
from fnd_ml.models.train_baseline import train_baseline
from fnd_ml.models.xgb import XgbConfig, train_xgb
from tests.fixtures.processed import two_dataset_corpus, write_processed
from tests.fixtures.synthetic import make_corpus

FITS: dict[str, Fit] = {
    "baseline": partial(baseline.fit_scorer, config=BaselineConfig(n_features=2**14)),
    "xgb": partial(
        xgb.fit_scorer, config=XgbConfig(n_features=2**12, svd_components=8, n_estimators=40)
    ),
}
TRAINERS: dict[str, Callable[..., dict[str, Any]]] = {
    "baseline": partial(train_baseline, config=BaselineConfig(n_features=2**14)),
    "xgb": partial(
        train_xgb, config=XgbConfig(n_features=2**12, svd_components=8, n_estimators=40)
    ),
}


@pytest.mark.parametrize("family", sorted(FITS))
def test_scorers_return_one_probability_per_row(family: str) -> None:
    frame = make_corpus(300, seed=4)
    train, val, calib, test = (
        frame.iloc[:160],
        frame.iloc[160:200],
        frame.iloc[200:250],
        frame.iloc[250:],
    )

    scores = FITS[family](train, val, calib)(test)

    assert scores.shape == (50,)
    assert scores.dtype == np.float64
    assert scores.min() >= 0.01
    assert scores.max() <= 0.99
    assert (
        scores[test["label"].to_numpy() == 1].mean() > scores[test["label"].to_numpy() == 0].mean()
    )


def test_a_loaded_baseline_scorer_reads_model_text_and_matches_the_saved_model(
    tmp_path: Path,
) -> None:
    frame = make_corpus(200, seed=6)
    texts = frame["model_text"].tolist()
    model = BaselineModel(BaselineConfig(n_features=2**14)).fit(texts, frame["label"])
    model.calibrate(texts, frame["label"])
    model.save(tmp_path / "artifact", {"version": "test-1"})

    scorer = baseline.load_scorer(tmp_path / "artifact")

    np.testing.assert_array_equal(scorer(frame), model.predict_proba(texts))
    # The scorer must read ``model_text`` (scrubbed), not the raw ``text`` column.
    disguised = frame.assign(text="unrelated words only")
    np.testing.assert_array_equal(scorer(disguised), model.predict_proba(texts))


@pytest.mark.parametrize("family", sorted(TRAINERS))
def test_training_refuses_a_changed_manifest(family: str, tmp_path: Path) -> None:
    processed, lock = write_processed(tmp_path, two_dataset_corpus())
    manifest = processed / MANIFEST_NAME
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ManifestMismatch, match="does not match the frozen"):
        TRAINERS[family](processed, tmp_path / "artifacts", tmp_path / "reports", lock_path=lock)

    assert not (tmp_path / "artifacts").exists()
    assert not (tmp_path / "reports").exists()


@pytest.mark.parametrize("family", sorted(TRAINERS))
def test_training_refuses_data_that_was_never_frozen(family: str, tmp_path: Path) -> None:
    processed, lock = write_processed(tmp_path, two_dataset_corpus(), freeze=False)

    with pytest.raises(ManifestMismatch, match="fnd data freeze"):
        TRAINERS[family](processed, tmp_path / "artifacts", tmp_path / "reports", lock_path=lock)
