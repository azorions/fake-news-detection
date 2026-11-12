import zipfile
from pathlib import Path

import pandas as pd
import pytest

from fnd_ml.data import loaders
from fnd_ml.data.schema import COLUMNS, validate_frame


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _liar_row(json_id: str, label: str, statement: str, speaker: str = "jane-doe") -> str:
    # 16 tab-separated columns, as in LIAR-PLUS: index, id, label, statement, subject, speaker, job,
    # state, party, five credit-history counts, context, justification
    cols = ["0", json_id, label, statement, "taxes", speaker, "Senator", "Ohio", "democrat"]
    cols += ["1", "0", "2", "0", "0", "a speech", "The record shows otherwise."]
    return "\t".join(cols)


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    _write(
        raw / "welfake" / "WELFake_Dataset.csv",
        ",title,text,label\n"
        '0,Senate passes bill,"WASHINGTON (Reuters) - The Senate voted on Tuesday.",0\n'
        "1,You won't believe this,Doctors hate this one trick that cures everything.,1\n"
        "2,Headline only,,1\n",
    )
    (raw / "isot").mkdir(parents=True)
    with zipfile.ZipFile(raw / "isot" / "News_dataset.zip", "w") as archive:
        archive.writestr(
            "Fake.csv",
            "title,text,subject,date\n"
            'Shocking truth,They lied to you about everything.,News,"December 31, 2017"\n',
        )
        archive.writestr(
            "True.csv",
            "title,text,subject,date\n"
            'Budget approved,"WASHINGTON (Reuters) - Lawmakers approved it.",politicsNews,"December 29, 2017 "\n',
        )
    _write(
        raw / "fakenewsnet" / "politifact_fake.csv",
        "id,news_url,title,tweet_ids\npf1,speedtalk.com/forum/viewtopic.php?t=1,Fake claim about a senator,1\t2\n",
    )
    _write(
        raw / "fakenewsnet" / "politifact_real.csv",
        "id,news_url,title,tweet_ids\npr1,https://www.nytimes.com/2017/a.html,Real report on the budget,3\n",
    )
    _write(
        raw / "fakenewsnet" / "gossipcop_fake.csv",
        "id,news_url,title,tweet_ids\ngf1,www.dailymail.co.uk/tvshowbiz/x.html,Star secretly married,4\n",
    )
    _write(
        raw / "fakenewsnet" / "gossipcop_real.csv",
        "id,news_url,title,tweet_ids\ngr1,,Star attends premiere,5\n",
    )
    liar_labels = ["true", "mostly-true", "half-true", "barely-true", "false", "pants-fire"]
    rows = [
        _liar_row(f"{i}.json", label, f"Statement number {i} about taxes.")
        for i, label in enumerate(liar_labels)
    ]
    _write(
        raw / "liar" / "train2.tsv", "\n".join(rows) + "\nbroken continuation line without tabs\n"
    )
    _write(
        raw / "liar" / "val2.tsv", _liar_row("90.json", "false", "Validation claim here.") + "\n"
    )
    _write(raw / "liar" / "test2.tsv", _liar_row("91.json", "true", "Test claim here.") + "\n")
    return raw


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
