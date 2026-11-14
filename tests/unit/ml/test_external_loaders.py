import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from fnd_ml.data import loaders
from fnd_ml.data.download import load_registry
from fnd_ml.data.schema import COLUMNS, validate_frame

REGISTRY = Path("data/registry/datasets.yaml")


# ---------------------------------------------------------------- fixtures


def _constraint(raw_dir: Path, test_rows: list[tuple[str, str, str]] | None = None) -> None:
    directory = raw_dir / "constraint"
    directory.mkdir(parents=True)
    files = {
        "Constraint_Train.csv": [
            ("1", "  States reported 734k tests and 532 deaths today.  ", "real"),
            ("2", 'A "miracle" drink cures the virus, doctors say,\nshare now', "fake"),
        ],
        "Constraint_Val.csv": [("1", "Masks reduce transmission in crowded rooms.", "real")],
        "english_test_with_labels.csv": test_rows
        or [("1", "5G towers spread the virus across the city.", "fake")],
    }
    for name, rows in files.items():
        frame = pd.DataFrame(rows, columns=["id", "tweet", "label"])
        frame.to_csv(directory / name, index=False)


def _article(veracity: str, portal: str, text: str, title: str = "A title") -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<article>\n'
        "  <author>Someone</author>\n"
        f"  <mainText>{text}</mainText>\n"
        "  <orientation>left</orientation>\n"
        f"  <portal>{portal}</portal>\n"
        f"  <title>{title}</title>\n"
        f"  <uri>http://{portal}.example/story</uri>\n"
        f"  <veracity>{veracity}</veracity>\n"
        "</article>\n"
    )


def _buzzfeed(raw_dir: Path) -> None:
    directory = raw_dir / "buzzfeed_webis"
    directory.mkdir(parents=True)
    articles = {
        "0001": _article("mostly true", "politico", "The senate passed the bill &amp; adjourned."),
        "0002": _article("mixture of true and false", "eagle-rising", "Half of this is right."),
        "0003": _article("mostly false", "occupy-democrats", "None of this happened."),
        "0004": _article("no factual content", "cnn", "An opinion about the debate."),
        "0005": _article("mostly true", "abc", ""),
    }
    with zipfile.ZipFile(directory / "articles.zip", "w") as archive:
        archive.writestr("articles/", "")
        for stem, document in articles.items():
            archive.writestr(f"articles/{stem}.xml", document)


def _satire(raw_dir: Path) -> None:
    directory = raw_dir / "satire_headlines"
    directory.mkdir(parents=True)
    records = [
        (1, "area man wins argument with mirror", "https://www.theonion.com/area-man-1"),
        (1, "nation's dogs vow to keep barking", "https://local.theonion.com/dogs-2"),
        (0, "senate passes budget after long debate", "https://www.huffingtonpost.com/entry/b_3"),
        (0, "what to watch this weekend", "https://www.huffingtonpost.comhttp://x.example/4"),
    ]
    lines = [
        json.dumps({"is_sarcastic": label, "headline": headline, "article_link": link})
        for label, headline, link in records
    ]
    (directory / "Sarcasm_Headlines_Dataset.json").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------- Constraint


def test_constraint_maps_fake_to_unreliable(tmp_path: Path) -> None:
    _constraint(tmp_path)

    frame = loaders.load_external("constraint", tmp_path)

    validate_frame(frame)
    assert list(frame.columns) == COLUMNS
    assert frame["label"].tolist() == [0, 1, 0, 1]
    assert frame["id"].tolist() == [
        "constraint:train:1",
        "constraint:train:2",
        "constraint:val:1",
        "constraint:test:1",
    ]
    assert frame.loc[0, "text"] == "States reported 734k tests and 532 deaths today."
    assert frame.loc[1, "text"].endswith("share now")  # quoted commas and line breaks survive
    assert set(frame["dataset"]) == {"constraint"}


def test_constraint_rejects_an_unknown_label(tmp_path: Path) -> None:
    _constraint(tmp_path, test_rows=[("1", "Some post about the virus.", "unsure")])

    with pytest.raises(ValueError, match="unsure"):
        loaders.load_external("constraint", tmp_path)


# ---------------------------------------------------------------- BuzzFeed-Webis


def test_buzzfeed_maps_ratings_and_drops_what_cannot_be_judged(tmp_path: Path) -> None:
    _buzzfeed(tmp_path)

    frame = loaders.load_external("buzzfeed_webis", tmp_path)

    validate_frame(frame)
    # 0004 has no factual content and 0005 has no text: neither can be scored.
    assert frame["id"].tolist() == [f"buzzfeed_webis:000{n}" for n in (1, 2, 3)]
    assert frame["label"].tolist() == [0, 1, 1]
    assert frame["publisher"].tolist() == ["politico", "eagle-rising", "occupy-democrats"]
    assert frame.loc[0, "text"] == "The senate passed the bill & adjourned."
    assert frame.loc[0, "title"] == "A title"
    assert frame.loc[0, "url"] == "http://politico.example/story"


# ---------------------------------------------------------------- satire headlines


def test_satire_label_means_satire(tmp_path: Path) -> None:
    _satire(tmp_path)

    frame = loaders.load_external("satire_headlines", tmp_path)

    validate_frame(frame)
    assert frame["label"].tolist() == [1, 1, 0, 0]
    assert frame["text"].tolist()[0] == "area man wins argument with mirror"
    assert frame["publisher"].tolist() == [
        "theonion.com",
        "theonion.com",
        "huffingtonpost.com",
        "huffingtonpost.com",
    ]
    assert frame["title"].isna().all()


# ---------------------------------------------------------------- these never reach training


def test_evaluation_sets_are_not_training_loaders() -> None:
    assert set(loaders.EXTERNAL_LOADERS) == {"constraint", "buzzfeed_webis", "satire_headlines"}
    assert not set(loaders.EXTERNAL_LOADERS) & set(loaders.LOADERS)


def test_training_loader_refuses_an_evaluation_set(tmp_path: Path) -> None:
    _constraint(tmp_path)

    with pytest.raises(KeyError):
        loaders.load("constraint", tmp_path)
    with pytest.raises(KeyError):
        loaders.load_external("welfake", tmp_path)


def test_registry_roles_match_the_loaders() -> None:
    registry = load_registry(REGISTRY)

    roles = {name: entry.role for name, entry in registry.items()}
    assert {name for name, role in roles.items() if role == "train"} == set(loaders.LOADERS)
    assert {name for name, role in roles.items() if role != "train"} == set(
        loaders.EXTERNAL_LOADERS
    )
    assert roles["satire_headlines"] == "satire"
    assert roles["constraint"] == roles["buzzfeed_webis"] == "external"
