import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score

from fnd_ml.common.artifacts import read_manifest, register, resolve
from fnd_ml.common.tokens import analyze, word_tokens
from fnd_ml.models.baseline import BaselineConfig, BaselineModel
from fnd_ml.models.train_baseline import train_baseline
from tests.fixtures.synthetic import NEUTRAL_TEXT, SENSATIONAL_TEXT, make_corpus

# ---------------------------------------------------------------- tokens


def test_word_tokens_are_lowercase_with_offsets_into_the_text() -> None:
    text = "SHOCKING: Doctors HATE it"
    tokens = word_tokens(text)
    assert [t[0] for t in tokens] == ["shocking", "doctors", "hate", "it"]
    assert all(text[start:end].lower() == token for token, start, end in tokens)


def test_analyze_yields_unigrams_then_bigrams() -> None:
    assert analyze("The Senate voted") == ["the", "senate", "voted", "the senate", "senate voted"]


def test_analyze_reads_only_the_first_max_tokens_words() -> None:
    assert analyze("one two three four five", max_tokens=2) == ["one", "two", "one two"]


# ---------------------------------------------------------------- model


@pytest.fixture(scope="module")
def corpus() -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
    frame = make_corpus(240, seed=1)
    train, held_out = frame.iloc[:180], frame.iloc[180:]
    return (
        train["model_text"].tolist(),
        train["label"].to_numpy(),
        held_out["model_text"].tolist(),
        held_out["label"].to_numpy(),
    )


@pytest.fixture(scope="module")
def model(corpus: tuple[list[str], np.ndarray, list[str], np.ndarray]) -> BaselineModel:
    train_texts, train_y, _, _ = corpus
    return BaselineModel(BaselineConfig(n_features=2**16)).fit(train_texts, train_y)


def test_baseline_separates_sensational_from_neutral_text(
    model: BaselineModel, corpus: tuple[list[str], np.ndarray, list[str], np.ndarray]
) -> None:
    _, _, texts, y = corpus
    predictions = (model.predict_raw(texts) >= 0.5).astype(int)
    assert f1_score(y, predictions, average="macro") > 0.95
    assert model.predict_raw([SENSATIONAL_TEXT])[0] > 0.8
    assert model.predict_raw([NEUTRAL_TEXT])[0] < 0.2


def test_fit_accepts_a_generator_of_texts(
    model: BaselineModel, corpus: tuple[list[str], np.ndarray, list[str], np.ndarray]
) -> None:
    train_texts, train_y, texts, _ = corpus
    streamed = BaselineModel(BaselineConfig(n_features=2**16)).fit(iter(train_texts), train_y)
    np.testing.assert_allclose(streamed.decision_function(texts), model.decision_function(texts))


def test_fit_rejects_mismatched_text_and_label_counts() -> None:
    with pytest.raises(ValueError, match="2 texts"):
        BaselineModel().fit(["one text here", "another text here"], np.array([0, 1, 1]))


def test_decision_is_the_linear_form_over_the_weighted_features(model: BaselineModel) -> None:
    texts = [SENSATIONAL_TEXT, NEUTRAL_TEXT, ""]
    expected = model.transform(texts) @ model.coef_ + model.intercept_
    np.testing.assert_allclose(model.decision_function(texts), expected)


def test_words_seen_in_fewer_than_min_df_documents_do_not_affect_the_score(
    model: BaselineModel,
) -> None:
    with_rare = model.decision_function([NEUTRAL_TEXT + " zzzneverseenbefore"])
    np.testing.assert_allclose(with_rare, model.decision_function([NEUTRAL_TEXT]))


def test_calibrated_probabilities_are_bounded_and_keep_the_ranking(
    model: BaselineModel, corpus: tuple[list[str], np.ndarray, list[str], np.ndarray]
) -> None:
    _, _, texts, y = corpus
    calibrated_model = BaselineModel(model.config).fit(*corpus[:2]).calibrate(texts, y)

    raw = calibrated_model.predict_raw(texts)
    calibrated = calibrated_model.predict_proba(texts)

    assert calibrated.min() >= 0.01
    assert calibrated.max() <= 0.99
    order = np.argsort(raw)
    assert (np.diff(calibrated[order]) >= 0).all()


def test_save_and_load_give_identical_predictions(
    model: BaselineModel,
    corpus: tuple[list[str], np.ndarray, list[str], np.ndarray],
    tmp_path: Path,
) -> None:
    _, _, texts, y = corpus
    fitted = BaselineModel(model.config).fit(*corpus[:2]).calibrate(texts, y)

    fitted.save(tmp_path / "artifact", {"version": "test-1"})
    loaded = BaselineModel.load(tmp_path / "artifact")

    np.testing.assert_array_equal(loaded.predict_proba(texts), fitted.predict_proba(texts))
    assert loaded.metadata["version"] == "test-1"
    assert sorted(p.name for p in (tmp_path / "artifact").iterdir()) == [
        "metadata.json",
        "model.npz",
    ]


def test_an_unfitted_model_refuses_to_predict() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        BaselineModel().predict_raw(["some text here"])


# ---------------------------------------------------------------- artifact registry


def test_registering_an_artifact_makes_it_resolvable(tmp_path: Path) -> None:
    assert read_manifest(tmp_path) == {}
    assert resolve(tmp_path, "baseline") is None

    (tmp_path / "baseline" / "v1").mkdir(parents=True)
    register(tmp_path, "baseline", "v1", "baseline/v1")

    assert read_manifest(tmp_path) == {"baseline": {"version": "v1", "path": "baseline/v1"}}
    assert resolve(tmp_path, "baseline") == tmp_path / "baseline" / "v1"


def test_a_registered_artifact_whose_files_are_gone_does_not_resolve(tmp_path: Path) -> None:
    register(tmp_path, "baseline", "v1", "baseline/v1")
    assert resolve(tmp_path, "baseline") is None


# ---------------------------------------------------------------- training command


def test_training_writes_an_artifact_a_registry_entry_and_a_report(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    (processed / "splits").mkdir(parents=True)
    frame = make_corpus(400, seed=2)
    bounds = {"train": (0, 240), "val": (240, 290), "calib": (290, 350), "test": (350, 400)}
    for split, (start, end) in bounds.items():
        frame.iloc[start:end].to_parquet(processed / "splits" / f"{split}.parquet", index=False)
    (processed / "splits_manifest.json").write_text('{"seed": 13}', encoding="utf-8")

    report = train_baseline(
        processed,
        tmp_path / "artifacts",
        tmp_path / "reports",
        BaselineConfig(n_features=2**16),
    )

    artifact = resolve(tmp_path / "artifacts", "baseline")
    assert artifact is not None
    assert {p.name for p in artifact.iterdir()} == {"model.npz", "metadata.json", "thresholds.json"}
    thresholds = json.loads((artifact / "thresholds.json").read_text(encoding="utf-8"))
    assert thresholds["t_low"] < thresholds["t_high"]
    assert report["version"] == artifact.name
    assert report["metrics"]["test"]["calibrated"]["macro_f1"] > 0.95
    assert set(report["metrics"]["test_by_dataset"]) == {"synthetic"}
    assert len(report["top_features"]["unreliable"]) > 0
    assert (tmp_path / "reports" / "baseline_v1.json").exists()
    assert "macro-F1" in (tmp_path / "reports" / "baseline_v1.md").read_text(encoding="utf-8")
