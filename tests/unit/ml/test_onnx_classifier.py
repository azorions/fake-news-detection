"""Serving the transformer through ONNX Runtime, on the committed toy artifact (no PyTorch)."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from fnd_ml.inference.onnx_classifier import OnnxClassifier
from tests.fixtures.synthetic import NEUTRAL_TEXT, SENSATIONAL_TEXT

FIXTURE = Path("tests/fixtures/tiny_transformer")


@pytest.fixture(scope="module")
def clf() -> OnnxClassifier:
    return OnnxClassifier.load(FIXTURE)


def test_the_artifact_names_itself(clf: OnnxClassifier) -> None:
    assert clf.version == "v1-tinyfix"
    assert clf.name == "tiny-fixture"
    assert clf.thresholds == (0.35, 0.65)
    assert clf.spec.max_length == 512


def test_predict_returns_a_calibrated_probability(clf: OnnxClassifier) -> None:
    out = clf.predict(None, SENSATIONAL_TEXT)

    assert out.version == "v1-tinyfix"
    assert 0.5 < out.p_unreliable <= 0.99
    assert not out.truncated
    # The artifact's temperature is 2: the served log-odds are half the raw ones.
    served, raw = (np.log(p / (1 - p)) for p in (out.p_unreliable, out.p_raw))
    assert served == pytest.approx(raw / 2, rel=1e-3)


def test_reporting_reads_as_reliable_and_unknown_words_as_undecided(clf: OnnxClassifier) -> None:
    assert clf.predict(None, NEUTRAL_TEXT).p_unreliable < 0.1
    assert clf.predict(None, "zebras quietly paint umbrellas").p_unreliable == pytest.approx(0.5)


def test_long_text_is_truncated_not_rejected(clf: OnnxClassifier) -> None:
    out = clf.predict("A title", "word " * 20000)  # about 100,000 characters

    assert out.truncated
    assert 0.01 <= out.p_unreliable <= 0.99


def test_three_words_without_a_title(clf: OnnxClassifier) -> None:
    out = clf.predict(None, "markets fell today")

    assert not out.truncated
    assert 0.01 <= out.p_unreliable <= 0.99


def test_the_title_is_read_too(clf: OnnxClassifier) -> None:
    body = "The committee met in the afternoon."

    with_title = clf.predict("SHOCKING hoax exposed", body).p_unreliable

    assert with_title > clf.predict(None, body).p_unreliable + 0.3


def test_a_batch_gives_the_same_answers_as_single_articles(clf: OnnxClassifier) -> None:
    articles = [
        (None, SENSATIONAL_TEXT),
        ("Budget approved", NEUTRAL_TEXT),
        (None, "markets fell today"),
        (float("nan"), "SHOCKING: the bank said the hoax was exposed " * 80),
    ]

    together = clf.probabilities(articles, batch_size=3)

    alone = [
        clf.predict(title if isinstance(title, str) else None, text) for title, text in articles
    ]
    np.testing.assert_allclose(together, [out.p_unreliable for out in alone], atol=1e-6)


def _copy(tmp_path: Path) -> Path:
    return Path(shutil.copytree(FIXTURE, tmp_path / "artifact"))


def test_damaged_model_file_raises_on_load(tmp_path: Path) -> None:
    artifact = _copy(tmp_path)
    model = artifact / "model.onnx"
    model.write_bytes(model.read_bytes()[:100])

    with pytest.raises(Exception, match=r"(?i)protobuf|onnx|model"):
        OnnxClassifier.load(artifact)


@pytest.mark.parametrize(
    "missing", ["metadata.json", "calibration.json", "thresholds.json", "tokenizer/tokenizer.json"]
)
def test_an_incomplete_artifact_raises_on_load(tmp_path: Path, missing: str) -> None:
    artifact = _copy(tmp_path)
    (artifact / missing).unlink()

    with pytest.raises(Exception):  # noqa: B017 - any failure to load must surface as an error
        OnnxClassifier.load(artifact)


def test_the_fixture_is_what_its_script_writes(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    from scripts.make_tiny_onnx_fixture import write_fixture

    fresh = OnnxClassifier.load(write_fixture(tmp_path / "fresh"))

    committed = OnnxClassifier.load(FIXTURE)
    assert (
        fresh.metadata
        == committed.metadata
        == json.loads((FIXTURE / "metadata.json").read_text(encoding="utf-8"))
    )
    for text in (SENSATIONAL_TEXT, NEUTRAL_TEXT, "markets fell today"):
        assert fresh.predict(None, text) == committed.predict(None, text)
