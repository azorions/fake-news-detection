from pathlib import Path

import pandas as pd
import pytest

from fnd_ml.data import loaders
from fnd_ml.data.schema import COLUMNS, validate_frame
from tests.fixtures.raw_datasets import make_raw_datasets


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    return make_raw_datasets(tmp_path)


@pytest.mark.parametrize("name", ["welfake", "isot", "fakenewsnet", "liar"])
def test_every_loader_returns_the_common_schema(raw_dir: Path, name: str) -> None:
    frame = loaders.load(name, raw_dir)
    assert list(frame.columns) == COLUMNS
    assert (frame["dataset"] == name).all()
    assert frame["id"].is_unique
    validate_frame(frame)


def test_welfake_keeps_its_labels_and_uses_the_title_when_the_body_is_empty(raw_dir: Path) -> None:
    frame = loaders.load("welfake", raw_dir)
    assert frame["label"].tolist() == [0, 1, 1]
    headline_only = frame.iloc[2]
    assert headline_only["text"] == "Headline only"
    assert pd.isna(headline_only["title"])


def test_isot_labels_fake_as_one_and_parses_dates(raw_dir: Path) -> None:
    frame = loaders.load("isot", raw_dir).set_index("title")
    assert frame.loc["Shocking truth", "label"] == 1
    assert frame.loc["Budget approved", "label"] == 0
    assert frame.loc["Budget approved", "date"] == "2017-12-29"


def test_fakenewsnet_uses_headlines_and_registrable_domains(raw_dir: Path) -> None:
    frame = loaders.load("fakenewsnet", raw_dir).set_index("text")
    assert frame.loc["Fake claim about a senator", "publisher"] == "speedtalk.com"
    assert frame.loc["Star secretly married", "publisher"] == "dailymail.co.uk"
    assert frame.loc["Real report on the budget", "label"] == 0
    assert frame.loc["Star secretly married", "label"] == 1
    assert pd.isna(frame.loc["Star attends premiere", "publisher"])


def test_liar_maps_the_six_ratings_and_drops_broken_rows(raw_dir: Path) -> None:
    frame = loaders.load("liar", raw_dir)
    by_text = dict(zip(frame["text"], frame["label"], strict=True))
    assert by_text["Statement number 0 about taxes."] == 0  # true
    assert by_text["Statement number 1 about taxes."] == 0  # mostly-true
    assert by_text["Statement number 4 about taxes."] == 1  # false
    assert by_text["Statement number 5 about taxes."] == 1  # pants-fire
    assert "Statement number 2 about taxes." not in by_text  # half-true is not binary
    assert len(frame) == 6  # 4 binary rows in train + val + test; the broken line is dropped
    assert set(frame["publisher"]) == {"jane-doe"}


def test_liar_middle_ratings_are_available_separately(raw_dir: Path) -> None:
    middle = loaders.load_liar_middle(raw_dir)
    assert sorted(middle["rating"]) == ["barely-true", "half-true"]
    assert list(middle.columns) == [*COLUMNS, "rating"]


def test_wire_label_check_reports_the_share_of_wire_copy_labelled_reliable(raw_dir: Path) -> None:
    assert loaders.wire_label_share(loaders.load("welfake", raw_dir)) == 1.0
    flipped = loaders.load("welfake", raw_dir).assign(label=lambda df: 1 - df["label"])
    assert loaders.wire_label_share(flipped) == 0.0


def test_validate_frame_rejects_labels_outside_zero_and_one(raw_dir: Path) -> None:
    frame = loaders.load("welfake", raw_dir)
    frame.loc[0, "label"] = 2
    with pytest.raises(ValueError, match="label"):
        validate_frame(frame)


def test_validate_frame_rejects_missing_text(raw_dir: Path) -> None:
    frame = loaders.load("welfake", raw_dir)
    frame.loc[0, "text"] = None
    with pytest.raises(ValueError, match="text"):
        validate_frame(frame)
