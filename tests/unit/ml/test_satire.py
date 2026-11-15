import itertools
import json
from pathlib import Path

import pandas as pd
import pytest

from fnd_ml.common.artifacts import resolve
from fnd_ml.common.text import text_hash
from fnd_ml.data.build import MANIFEST_NAME, ManifestMismatch, write_lock
from fnd_ml.inference.satire_scorer import HEADLINE_WORDS, SatireScorer
from fnd_ml.models.baseline import BaselineConfig
from fnd_ml.models.satire import satire_split, train_satire

_VERBS = ["wins", "loses", "regrets", "explains", "forgets", "celebrates", "ignores", "rates"]
_OBJECTS = ["argument", "lunch order", "own birthday", "parking spot", "group chat", "the weekend"]
_BODIES = ["City council", "Health ministry", "Transit agency", "School board", "Central bank"]
_ACTIONS = ["approves", "delays", "reviews", "publishes", "extends", "announces"]
_TOPICS = ["annual budget", "road repairs", "flu guidance", "exam schedule", "rate decision"]
CONFIG = BaselineConfig(n_features=2**14, max_tokens=64, min_df=2, C=2.0)


def _headlines() -> pd.DataFrame:
    """Satire headlines start with "Area man"; plain ones report what an institution did."""
    satire = [f"Area man {v} {o}" for v, o in itertools.product(_VERBS, _OBJECTS)]
    plain = [f"{b} {a} {t}" for b, a, t in itertools.product(_BODIES, _ACTIONS, _TOPICS)][:60]
    texts = satire + plain
    return pd.DataFrame(
        {
            "id": [f"satire_headlines:{i}" for i in range(len(texts))],
            "dataset": "satire_headlines",
            "title": None,
            "text": texts,
            "label": [1] * len(satire) + [0] * len(plain),
            "publisher": ["satire.example"] * len(satire) + ["news.example"] * len(plain),
            "model_text": texts,
            "text_hash": [text_hash(text) for text in texts],
        }
    )


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, object]]:
    root = tmp_path_factory.mktemp("satire")
    processed = root / "processed"
    (processed / "eval").mkdir(parents=True)
    _headlines().to_parquet(processed / "eval" / "satire_headlines.parquet", index=False)
    (processed / MANIFEST_NAME).write_text(json.dumps({"rows": {"total": 0}}), encoding="utf-8")
    write_lock(processed, root / "data.lock")
    report = train_satire(
        processed, root / "artifacts", root / "reports", CONFIG, lock_path=root / "data.lock"
    )
    return root, report


def test_the_split_depends_only_on_the_text_hash() -> None:
    frame = _headlines()
    twice = pd.concat([frame, frame], ignore_index=True)

    split = satire_split(twice["text_hash"])

    assert set(split) == {"train", "calib", "test"}
    assert (split.iloc[: len(frame)].to_numpy() == split.iloc[len(frame) :].to_numpy()).all()
    shares = split.value_counts(normalize=True)
    assert 0.65 < shares["train"] < 0.92


def test_the_test_tenth_is_written_and_is_disjoint_from_train(
    trained: tuple[Path, dict[str, object]],
) -> None:
    root, report = trained
    frame = _headlines()
    split = satire_split(frame["text_hash"])

    held_out = pd.read_parquet(root / "processed" / "eval" / "satire_slice.parquet")

    assert set(held_out["text_hash"]) == set(frame.loc[split == "test", "text_hash"])
    assert not set(held_out["text_hash"]) & set(frame.loc[split != "test", "text_hash"])
    assert report["rows"] == {
        "train": int((split == "train").sum()),
        "calib": int((split == "calib").sum()),
        "test": len(held_out),
    }


def test_training_registers_a_satire_artifact_and_reports_roc_auc_on_the_test_tenth(
    trained: tuple[Path, dict[str, object]],
) -> None:
    root, report = trained

    artifact = resolve(root / "artifacts", "satire")

    assert artifact is not None
    assert report["version"] == artifact.name
    assert json.loads((artifact / "metadata.json").read_text(encoding="utf-8"))["kind"] == "satire"
    assert resolve(root / "artifacts", "baseline") is None
    assert report["metrics"]["test"]["calibrated"]["roc_auc"] > 0.95  # type: ignore[index]
    markdown = (root / "reports" / "satire_v2.md").read_text(encoding="utf-8")
    assert "ROC-AUC" in markdown
    assert "house style" in markdown
    assert "satire.example" in markdown and "news.example" in markdown


def test_a_headline_in_the_satire_pattern_scores_high_and_a_plain_one_low(
    trained: tuple[Path, dict[str, object]],
) -> None:
    root, report = trained
    artifact = resolve(root / "artifacts", "satire")
    assert artifact is not None

    scorer = SatireScorer.load(artifact)

    assert scorer.version == report["version"]
    assert scorer.score("Area man wins argument", "The body is not read.") > 0.8
    assert scorer.score("City council approves annual budget", "The body is not read.") < 0.2


def test_the_title_is_scored_when_there_is_one(trained: tuple[Path, dict[str, object]]) -> None:
    root, _ = trained
    artifact = resolve(root / "artifacts", "satire")
    assert artifact is not None
    scorer = SatireScorer.load(artifact)

    with_plain_body = scorer.score("Area man wins argument", "City council approves annual budget")
    with_satire_body = scorer.score("Area man wins argument", "Area man loses lunch order")

    assert with_plain_body == with_satire_body


def test_without_a_title_only_the_first_40_words_are_read(
    trained: tuple[Path, dict[str, object]],
) -> None:
    root, _ = trained
    artifact = resolve(root / "artifacts", "satire")
    assert artifact is not None
    scorer = SatireScorer.load(artifact)
    assert HEADLINE_WORDS == 40
    opening = "Area man wins argument " + "and then some more " * 9  # 4 + 36 = 40 words
    assert len(opening.split()) == 40

    base = scorer.score(None, opening)

    assert scorer.score(None, opening + "council approves annual budget") == base
    assert scorer.score("", opening + "council approves annual budget") == base
    # Word 40 is read: replacing it changes the score.
    changed = " ".join([*opening.split()[:39], "budget"])
    assert scorer.score(None, changed) != base
    assert base > 0.5


def test_training_satire_refuses_a_changed_manifest(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    (processed / "eval").mkdir(parents=True)
    _headlines().to_parquet(processed / "eval" / "satire_headlines.parquet", index=False)
    (processed / MANIFEST_NAME).write_text(json.dumps({"rows": {"total": 0}}), encoding="utf-8")
    write_lock(processed, tmp_path / "data.lock")
    (processed / MANIFEST_NAME).write_text(json.dumps({"rows": {"total": 1}}), encoding="utf-8")

    with pytest.raises(ManifestMismatch):
        train_satire(
            processed,
            tmp_path / "artifacts",
            tmp_path / "reports",
            CONFIG,
            lock_path=tmp_path / "l",
        )
    with pytest.raises(ManifestMismatch, match="does not match the frozen"):
        train_satire(
            processed,
            tmp_path / "artifacts",
            tmp_path / "reports",
            CONFIG,
            lock_path=tmp_path / "data.lock",
        )
    assert not (processed / "eval" / "satire_slice.parquet").exists()
