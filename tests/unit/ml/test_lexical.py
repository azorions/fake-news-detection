from pathlib import Path

import numpy as np
import pytest

from fnd_ml.common.text import model_input
from fnd_ml.explain.lexical import TokenWeight, attribute, total_logit
from fnd_ml.inference.baseline_classifier import BaselineClassifier, ClassifierOutput
from fnd_ml.models.baseline import BaselineConfig, BaselineModel
from tests.fixtures.synthetic import NEUTRAL_TEXT, SENSATIONAL_TEXT, make_corpus

TITLE = "BOMBSHELL: big pharma covers up a miracle cure"
DATELINE = "WASHINGTON (Reuters) - "


@pytest.fixture(scope="module")
def clf(tmp_path_factory: pytest.TempPathFactory) -> BaselineClassifier:
    frame = make_corpus(240, seed=1)
    texts, y = frame["model_text"].tolist(), frame["label"].to_numpy()
    model = BaselineModel(BaselineConfig(n_features=2**16)).fit(texts, y).calibrate(texts, y)
    directory: Path = tmp_path_factory.mktemp("artifact") / "v-test"
    model.save(directory, {"version": "v-test"})
    (directory / "thresholds.json").write_text('{"t_low": 0.3, "t_high": 0.7}', encoding="utf-8")
    return BaselineClassifier.load(directory)


# ---------------------------------------------------------------- classifier


def test_classifier_reports_calibrated_and_raw_probability_with_its_version(
    clf: BaselineClassifier,
) -> None:
    risky = clf.predict(None, SENSATIONAL_TEXT)
    safe = clf.predict(None, NEUTRAL_TEXT)
    assert isinstance(risky, ClassifierOutput)
    assert risky.p_unreliable > 0.8 > 0.2 > safe.p_unreliable
    assert 0.0 < risky.p_raw < 1.0
    assert risky.version == "v-test"
    assert risky.truncated is False
    assert clf.thresholds == (0.3, 0.7)


def test_classifier_reads_the_title_together_with_the_text(clf: BaselineClassifier) -> None:
    with_title = clf.predict(TITLE, NEUTRAL_TEXT)
    assert with_title.p_raw > clf.predict(None, NEUTRAL_TEXT).p_raw


def test_classifier_flags_text_longer_than_the_model_reads(clf: BaselineClassifier) -> None:
    limit = clf.model.config.max_tokens
    assert clf.predict(None, "word " * limit).truncated is False
    assert clf.predict(None, "word " * (limit + 1)).truncated is True


def test_classifier_falls_back_to_default_thresholds(tmp_path: Path) -> None:
    frame = make_corpus(40)
    model = BaselineModel(BaselineConfig(n_features=2**12)).fit(frame["model_text"], frame["label"])
    model.save(tmp_path, {})
    loaded = BaselineClassifier.load(tmp_path)
    assert loaded.thresholds == (0.35, 0.65)
    assert loaded.version == tmp_path.name


# ---------------------------------------------------------------- attribution


@pytest.mark.parametrize(
    ("title", "text"),
    [
        (None, SENSATIONAL_TEXT),
        (None, NEUTRAL_TEXT),
        (TITLE, NEUTRAL_TEXT + " " + SENSATIONAL_TEXT),
        (TITLE, DATELINE + NEUTRAL_TEXT),
        (None, "the truth " * 700),
    ],
)
def test_token_weights_plus_intercept_equal_the_decision_function(
    clf: BaselineClassifier, title: str | None, text: str
) -> None:
    expected = clf.model.decision_function([model_input(title, text)[0]])[0]
    assert total_logit(clf, title, text) == pytest.approx(expected, abs=1e-6)


def test_weights_of_a_text_without_title_sum_to_the_logit(clf: BaselineClassifier) -> None:
    weights = attribute(clf, None, SENSATIONAL_TEXT, top_k=None)
    total = sum(w.weight for w in weights) + clf.model.intercept_
    assert total == pytest.approx(total_logit(clf, None, SENSATIONAL_TEXT), abs=1e-6)


def test_spans_point_at_their_token_in_the_original_text(clf: BaselineClassifier) -> None:
    text = NEUTRAL_TEXT + " " + SENSATIONAL_TEXT
    weights = attribute(clf, TITLE, text, top_k=None)
    assert weights
    assert all(isinstance(w, TokenWeight) for w in weights)
    assert all(text[w.start : w.end].lower() == w.token for w in weights)
    assert [w.start for w in weights] == sorted(w.start for w in weights)


def test_sensational_words_push_toward_unreliable(clf: BaselineClassifier) -> None:
    by_token = {w.token: w.weight for w in attribute(clf, None, SENSATIONAL_TEXT, top_k=None)}
    assert by_token["shocking"] > 0
    neutral = {w.token: w.weight for w in attribute(clf, None, NEUTRAL_TEXT, top_k=None)}
    assert neutral["ministry"] < 0


def test_offsets_stay_correct_when_artifacts_are_removed(clf: BaselineClassifier) -> None:
    text = (
        DATELINE + "SHOCKING: the government (Source: example) is hiding the truth about vaccines!"
    )
    weights = attribute(clf, None, text, top_k=None)
    assert all(text[w.start : w.end].lower() == w.token for w in weights)
    assert min(w.start for w in weights) >= len(DATELINE)
    assert "reuters" not in {w.token for w in weights}
    assert "shocking" in {w.token for w in weights}


def test_title_tokens_are_left_out_of_the_spans(clf: BaselineClassifier) -> None:
    weights = attribute(clf, TITLE, NEUTRAL_TEXT, top_k=None)
    assert "bombshell" not in {w.token for w in weights}
    in_text = sum(w.weight for w in weights) + clf.model.intercept_
    assert in_text < total_logit(clf, TITLE, NEUTRAL_TEXT)


def test_top_k_keeps_the_strongest_tokens(clf: BaselineClassifier) -> None:
    everything = attribute(clf, None, SENSATIONAL_TEXT, top_k=None)
    top = attribute(clf, None, SENSATIONAL_TEXT, top_k=3)
    assert len(top) == 3
    weakest_kept = min(abs(w.weight) for w in top)
    dropped = [w for w in everything if w not in top]
    assert all(abs(w.weight) <= weakest_kept for w in dropped)
    assert len(attribute(clf, None, SENSATIONAL_TEXT)) <= 40


def test_out_of_vocabulary_text_has_no_attribution(clf: BaselineClassifier) -> None:
    assert attribute(clf, None, "zxqv wklp mnbq") == []
    assert attribute(clf, None, "") == []
    assert total_logit(clf, None, "zxqv wklp mnbq") == pytest.approx(clf.model.intercept_)
    assert np.isfinite(clf.predict(None, "").p_unreliable)
