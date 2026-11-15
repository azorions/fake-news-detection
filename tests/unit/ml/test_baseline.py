import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score

from fnd_ml.common.artifacts import read_manifest, register, resolve
from fnd_ml.common.tokens import analyze, analyze_chars, word_tokens
from fnd_ml.models.baseline import BaselineConfig, BaselineModel
from fnd_ml.models.train_baseline import ablation_char_ngrams, stratified_positions, train_baseline
from tests.fixtures.processed import two_dataset_corpus, write_processed
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


def _identifiers(tmp_path: Path, *terms: str) -> Path:
    path = tmp_path / "identifiers.txt"
    path.write_text("\n".join(terms) + "\n", encoding="utf-8")
    return path


def test_training_writes_an_artifact_a_registry_entry_and_a_report(tmp_path: Path) -> None:
    processed, lock = write_processed(tmp_path, two_dataset_corpus())

    report = train_baseline(
        processed,
        tmp_path / "artifacts",
        tmp_path / "reports",
        BaselineConfig(n_features=2**16),
        lock_path=lock,
        identifiers_path=_identifiers(tmp_path, "reuters"),
    )

    artifact = resolve(tmp_path / "artifacts", "baseline")
    assert artifact is not None
    assert {p.name for p in artifact.iterdir()} == {"model.npz", "metadata.json", "thresholds.json"}
    thresholds = json.loads((artifact / "thresholds.json").read_text(encoding="utf-8"))
    assert thresholds["t_low"] < thresholds["t_high"]
    assert report["version"] == artifact.name
    assert report["version"].startswith("v2-")
    assert report["metrics"]["test"]["calibrated"]["macro_f1"] > 0.95
    assert set(report["metrics"]["test_by_dataset"]) == {"alpha", "beta"}
    assert len(report["top_features"]["unreliable"]) > 0
    assert report["leak_probe"] == {"flagged": [], "unnamed_buckets": 0}
    # Both datasets are written by the same generator, so the text does not give the dataset away.
    assert report["origin_probe"]["accuracy"] < 0.75
    assert set(report["origin_probe"]["per_label"]) == {"0", "1"}
    assert (tmp_path / "reports" / "baseline_v2.json").exists()
    markdown = (tmp_path / "reports" / "baseline_v2.md").read_text(encoding="utf-8")
    assert "macro-F1" in markdown
    assert "## Leak probes" in markdown


def test_training_never_opens_the_test_split_when_a_top_feature_names_a_source(
    tmp_path: Path,
) -> None:
    processed, lock = write_processed(tmp_path, two_dataset_corpus())
    (processed / "splits" / "test.parquet").unlink()  # any read of the test split would now fail

    report = train_baseline(
        processed,
        tmp_path / "artifacts",
        tmp_path / "reports",
        BaselineConfig(n_features=2**16),
        lock_path=lock,
        identifiers_path=_identifiers(tmp_path, "shocking"),
    )

    assert "shocking" in report["leak_probe"]["flagged"]
    assert "test" not in report["metrics"]
    assert "bands_on_test" not in report
    assert resolve(tmp_path / "artifacts", "baseline") is None
    markdown = (tmp_path / "reports" / "baseline_v2.md").read_text(encoding="utf-8")
    assert "`shocking`" in markdown
    assert "not scored" in markdown


def test_training_on_a_single_dataset_has_no_origin_probe(tmp_path: Path) -> None:
    processed, lock = write_processed(tmp_path, make_corpus(400, seed=2))

    report = train_baseline(
        processed,
        tmp_path / "artifacts",
        tmp_path / "reports",
        BaselineConfig(n_features=2**16),
        lock_path=lock,
        identifiers_path=_identifiers(tmp_path, "reuters"),
    )

    assert report["origin_probe"] is None
    assert set(report["metrics"]["test_by_dataset"]) == {"synthetic"}


def test_a_bucket_is_named_by_an_n_gram_seen_once_in_the_naming_sample(
    model: BaselineModel, corpus: tuple[list[str], np.ndarray, list[str], np.ndarray]
) -> None:
    train_texts, _, _, _ = corpus
    strongest = model.top_features(train_texts, k=1)["unreliable"][0]
    one_text = next(text for text in train_texts if analyze(text).count(strongest) == 1)

    assert model.top_features([one_text], k=1)["unreliable"] == [strongest]


# ---------------------------------------------------------------- character n-grams


def test_char_wb_analysis_adds_3_to_5_character_grams_inside_word_boundaries() -> None:
    assert analyze_chars("Go now") == [
        " go",
        "go ",
        " go ",
        " no",
        "now",
        "ow ",
        " now",
        "now ",
        " now ",
    ]


def test_character_n_grams_let_an_unseen_word_form_change_the_score(
    corpus: tuple[list[str], np.ndarray, list[str], np.ndarray],
) -> None:
    train_texts, train_y, _, _ = corpus
    words_only = BaselineModel(BaselineConfig(n_features=2**16)).fit(train_texts, train_y)
    with_chars = BaselineModel(BaselineConfig(n_features=2**16, char_ngrams=True)).fit(
        train_texts, train_y
    )
    unseen_form = NEUTRAL_TEXT + " shockingly"

    np.testing.assert_allclose(
        words_only.decision_function([unseen_form]), words_only.decision_function([NEUTRAL_TEXT])
    )
    assert (
        with_chars.decision_function([unseen_form])[0]
        > with_chars.decision_function([NEUTRAL_TEXT])[0] + 0.01
    )


def test_a_model_with_character_n_grams_survives_save_and_load(
    corpus: tuple[list[str], np.ndarray, list[str], np.ndarray], tmp_path: Path
) -> None:
    train_texts, train_y, texts, _ = corpus
    fitted = BaselineModel(BaselineConfig(n_features=2**16, char_ngrams=True)).fit(
        train_texts, train_y
    )

    fitted.save(tmp_path / "artifact", {"version": "test-chars"})
    loaded = BaselineModel.load(tmp_path / "artifact")

    np.testing.assert_array_equal(loaded.predict_raw(texts), fitted.predict_raw(texts))


def test_the_ablation_compares_both_feature_sets_on_val_and_writes_a_report(
    tmp_path: Path,
) -> None:
    processed, lock = write_processed(tmp_path, two_dataset_corpus())

    result = ablation_char_ngrams(
        processed,
        tmp_path / "reports",
        BaselineConfig(n_features=2**16),
        sample_rows=120,
        lock_path=lock,
    )

    assert [variant["name"] for variant in result["variants"]] == [
        "word 1-2",
        "word 1-2 + char_wb 3-5",
    ]
    for variant in result["variants"]:
        assert variant["val"]["macro_f1"] > 0.9
        assert variant["val"]["roc_auc"] > 0.9
        assert variant["peak_memory_mb"] >= 0
    assert result["rows"] == {"train_sample": 120, "train": 240, "val": 50}
    # Both feature sets separate this corpus, so the gain is far below the 0.01 needed to adopt.
    assert result["adopt_char_ngrams"] is False
    markdown = (tmp_path / "reports" / "ablation_char_ngrams.md").read_text(encoding="utf-8")
    assert "word 1-2 + char_wb 3-5" in markdown
    assert "Decision" in markdown
    assert not (tmp_path / "artifacts").exists()


def test_the_ablation_sample_keeps_the_label_and_dataset_mix_of_train() -> None:
    frame = make_corpus(400, seed=2)  # labels alternate
    frame["dataset"] = np.where(np.arange(400) < 300, "alpha", "beta")  # 3 of 4 rows are alpha

    chosen = stratified_positions(frame[["dataset", "label"]], 200, seed=13)

    sample = frame.iloc[chosen]
    assert len(set(chosen.tolist())) == 200
    assert sample["dataset"].value_counts().to_dict() == {"alpha": 150, "beta": 50}
    assert sample["label"].value_counts().to_dict() == {0: 100, 1: 100}
    # A size the groups cannot share evenly is still met exactly.
    assert len(stratified_positions(frame[["dataset", "label"]], 101, seed=13)) == 101
    assert len(stratified_positions(frame[["dataset", "label"]], 5000, seed=13)) == 400
