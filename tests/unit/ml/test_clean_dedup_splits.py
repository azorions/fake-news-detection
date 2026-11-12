import json
from pathlib import Path

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from fnd_ml.data.build import build_corpus
from fnd_ml.data.clean import clean_frame
from fnd_ml.data.dedup import drop_exact_duplicates
from fnd_ml.data.schema import COLUMNS
from fnd_ml.data.splits import SPLITS, assign_group, assign_split
from tests.fixtures.raw_datasets import make_raw_datasets


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    base = {"title": None, "publisher": None, "url": None, "date": None, "dataset": "toy"}
    frame = pd.DataFrame([{**base, "id": f"toy:{i}", **row} for i, row in enumerate(rows)])
    return frame[COLUMNS]


# ---------------------------------------------------------------- clean


def test_clean_adds_model_text_word_counts_buckets_and_hashes() -> None:
    article = " ".join(["word"] * 200)
    frame = _frame(
        [
            {"title": "Budget passes", "text": "WASHINGTON (Reuters) - " + article, "label": 0},
            {"text": "Ten words in this headline about the senate vote today", "label": 1},
            {"text": "Short   text “quoted” with fifty " + " ".join(["w"] * 50), "label": 1},
        ]
    )

    cleaned = clean_frame(frame)

    assert cleaned["length_bucket"].tolist() == ["article", "headline", "short"]
    assert cleaned.loc[0, "model_text"] == "Budget passes\n\n" + article
    assert "Reuters" in cleaned.loc[0, "text"]  # stored text keeps the dateline
    assert '"quoted"' in cleaned.loc[2, "text"]  # normalized
    assert cleaned.loc[1, "n_words"] == 10
    assert cleaned["text_hash"].str.len().eq(64).all()


def test_clean_drops_rows_with_fewer_than_three_words() -> None:
    frame = _frame([{"text": "Too short", "label": 0}, {"text": "Long enough text", "label": 1}])
    assert clean_frame(frame)["text"].tolist() == ["Long enough text"]


def test_rows_that_are_only_publisher_artifacts_are_dropped() -> None:
    frame = _frame([{"text": "WASHINGTON (Reuters) - https://t.co/x", "label": 0}])
    assert clean_frame(frame).empty


# ---------------------------------------------------------------- dedup


def test_exact_duplicates_ignoring_case_punctuation_and_datelines_collapse() -> None:
    frame = clean_frame(
        _frame(
            [
                {"text": "WASHINGTON (Reuters) - The Senate voted today.", "label": 0},
                {"text": "the senate voted today", "label": 0},
                {"text": "Something else entirely here", "label": 1},
            ]
        )
    )

    deduped, stats = drop_exact_duplicates(frame)

    assert len(deduped) == 2
    assert stats == {"input": 3, "duplicates_removed": 1, "conflicting_removed": 0, "output": 2}


def test_duplicates_with_conflicting_labels_are_removed_entirely() -> None:
    frame = clean_frame(
        _frame(
            [
                {"text": "The same words appear here", "label": 0},
                {"text": "The same words appear here!", "label": 1},
                {"text": "A different text altogether", "label": 1},
            ]
        )
    )

    deduped, stats = drop_exact_duplicates(frame)

    assert deduped["text"].tolist() == ["A different text altogether"]
    assert stats["conflicting_removed"] == 2


# ---------------------------------------------------------------- groups and splits


def test_groups_use_domain_then_dataset_publisher_then_row_hash() -> None:
    frame = clean_frame(
        _frame(
            [
                {
                    "text": "Headline from a site",
                    "label": 1,
                    "url": "x.com/a",
                    "publisher": "x.com",
                },
                {"text": "Statement by a speaker", "label": 0, "publisher": "jane-doe"},
                {"text": "Article without a source", "label": 0},
            ]
        )
    )

    groups = assign_group(frame)

    assert groups.tolist()[:2] == ["domain:x.com", "toy:jane-doe"]
    assert groups.tolist()[2] == "row:" + frame.loc[2, "text_hash"]


@given(st.lists(st.text(min_size=1, max_size=8), min_size=1, max_size=300))
def test_split_assignment_is_deterministic_and_never_splits_a_group(groups: list[str]) -> None:
    series = pd.Series(groups)
    first = assign_split(series, seed=13)
    assert first.equals(assign_split(series, seed=13))
    assert set(first) <= set(SPLITS)
    assert (pd.DataFrame({"g": series, "s": first}).groupby("g")["s"].nunique() == 1).all()


def test_split_proportions_are_roughly_70_10_10_10() -> None:
    shares = assign_split(pd.Series([f"row:{i}" for i in range(20_000)])).value_counts(
        normalize=True
    )
    assert shares["train"] == pytest.approx(0.70, abs=0.02)
    for split in ("val", "calib", "test"):
        assert shares[split] == pytest.approx(0.10, abs=0.02)


# ---------------------------------------------------------------- build


@pytest.fixture
def tiny_raw(tmp_path: Path) -> Path:
    return make_raw_datasets(tmp_path)


def test_build_writes_corpus_splits_and_a_manifest(tiny_raw: Path, tmp_path: Path) -> None:
    out = tmp_path / "processed"

    manifest = build_corpus(tiny_raw, out, seed=13)

    assert (out / "corpus.parquet").exists()
    assert (out / "eval" / "liar_middle.parquet").exists()
    for split in SPLITS:
        assert (out / "splits" / f"{split}.parquet").exists()
    on_disk = json.loads((out / "splits_manifest.json").read_text(encoding="utf-8"))
    assert on_disk == manifest
    assert manifest["seed"] == 13
    assert manifest["shared_groups_between_splits"] == 0
    assert manifest["shared_hashes_between_splits"] == 0
    assert sum(manifest["rows"][s] for s in SPLITS) == manifest["rows"]["total"]
    assert set(manifest["files"]) >= {"corpus.parquet", "splits/train.parquet"}


def test_build_refuses_an_inverted_wire_label_map(
    tiny_raw: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fnd_ml.data import build, loaders

    real_load = loaders.load

    def flipped(name: str, raw_dir: Path) -> pd.DataFrame:
        frame = real_load(name, raw_dir)
        return frame.assign(label=1 - frame["label"]) if name == "welfake" else frame

    monkeypatch.setattr(build.loaders, "load", flipped)
    with pytest.raises(ValueError, match="welfake"):
        build_corpus(tiny_raw, tmp_path / "processed")
