import json
from pathlib import Path

import pandas as pd
import pytest

from fnd_ml.data.build import ManifestMismatch, assert_frozen, build_corpus, write_lock
from fnd_ml.data.clean import add_year
from tests.fixtures.raw_datasets import make_raw_datasets

# English-looking text: the cleaner drops long texts without common English words.
ARTICLE = " ".join(
    f"the committee reviewed item {i} and it was approved by the board" for i in range(10, 40)
)
ARTICLE_EDITED = ARTICLE.replace("item 20 ", "point 20 ")  # one word changed
ISOT_TEXT = "WASHINGTON (Reuters) - Lawmakers approved it."  # dated 2017-12-29 in the ISOT fixture
TRICK = "Doctors hate this one trick that cures everything."


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    """The four tiny training sets, plus: a near-duplicate pair, a WELFake copy of a dated ISOT
    article, and two of the three evaluation sets (BuzzFeed-Webis is left out on purpose)."""
    raw = make_raw_datasets(tmp_path)
    (raw / "welfake" / "WELFake_Dataset.csv").write_text(
        ",title,text,label\n"
        f"0,Long report,{ARTICLE},0\n"
        f"1,Long report again,{ARTICLE_EDITED},0\n"
        f'2,Budget,"{ISOT_TEXT}",0\n'
        f"3,You won't believe this,{TRICK},1\n",
        encoding="utf-8",
    )
    constraint = raw / "constraint"
    constraint.mkdir()
    rows = {
        "Constraint_Train.csv": [("1", TRICK, "fake"), ("2", "Masks reduce transmission.", "real")],
        "Constraint_Val.csv": [("1", "Hospitals reported fewer cases this week.", "real")],
        "english_test_with_labels.csv": [("1", "5G towers spread the virus here.", "fake")],
    }
    for name, content in rows.items():
        pd.DataFrame(content, columns=["id", "tweet", "label"]).to_csv(
            constraint / name, index=False
        )
    satire = raw / "satire_headlines"
    satire.mkdir()
    lines = [
        {"is_sarcastic": 1, "headline": "area man wins argument with mirror", "article_link": "o"},
        {"is_sarcastic": 0, "headline": "senate passes budget after debate", "article_link": "h"},
    ]
    (satire / "Sarcasm_Headlines_Dataset.json").write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8"
    )
    return raw


def test_build_removes_near_duplicates_and_reports_hygiene(raw: Path, tmp_path: Path) -> None:
    out = tmp_path / "processed"

    manifest = build_corpus(raw, out)

    corpus = pd.read_parquet(out / "corpus.parquet")
    assert manifest["splits_version"] == 2
    assert manifest["near_dedup"]["near_duplicates_removed"] == 1
    assert corpus["text"].str.startswith("the committee reviewed").sum() == 1
    assert manifest["shared_groups_between_splits"] == 0
    assert manifest["shared_hashes_between_splits"] == 0
    assert manifest["shared_near_duplicate_clusters_between_splits"] == 0
    assert manifest["rows"]["total"] == len(corpus)


def test_a_duplicate_keeps_the_date_of_its_dated_copy(raw: Path, tmp_path: Path) -> None:
    out = tmp_path / "processed"

    manifest = build_corpus(raw, out)

    corpus = pd.read_parquet(out / "corpus.parquet")
    row = corpus[corpus["text"] == ISOT_TEXT].iloc[0]
    assert row["dataset"] == "welfake"
    assert row["also_in"] == "isot"
    assert row["year"] == 2017
    assert str(corpus["year"].dtype) == "Int16"
    assert manifest["dates"]["min_year"] == manifest["dates"]["max_year"] == 2017
    assert manifest["dates"]["rows_with_year"] == int(corpus["year"].notna().sum()) >= 1


def test_evaluation_sets_are_written_without_rows_that_are_in_the_corpus(
    raw: Path, tmp_path: Path
) -> None:
    out = tmp_path / "processed"

    manifest = build_corpus(raw, out)

    constraint = pd.read_parquet(out / "eval" / "constraint.parquet")
    corpus = pd.read_parquet(out / "corpus.parquet")
    assert manifest["external"]["constraint"] == {
        "rows": 3,
        "unreliable_share": round(1 / 3, 4),
        "dropped_overlap_with_corpus": 1,
    }
    assert TRICK not in set(constraint["text"])
    assert not set(constraint["text_hash"]) & set(corpus["text_hash"])
    assert {"model_text", "length_bucket", "year"} <= set(constraint.columns)
    assert manifest["external"]["satire_headlines"]["rows"] == 2
    assert not set(corpus["dataset"]) & {"constraint", "satire_headlines"}
    assert manifest["external_missing"] == ["buzzfeed_webis"]
    assert {"eval/constraint.parquet", "eval/satire_headlines.parquet"} <= set(manifest["files"])


def test_add_year_reads_dates_and_leaves_gaps_empty() -> None:
    frame = pd.DataFrame({"date": ["2017-12-29", None, "not a date", "2016-01-05"]})

    years = add_year(frame)["year"]

    assert years.tolist()[0] == 2017 and years.tolist()[3] == 2016
    assert years.isna().tolist() == [False, True, True, False]


# ---------------------------------------------------------------- freezing


def test_frozen_manifest_is_accepted_and_a_changed_one_refused(raw: Path, tmp_path: Path) -> None:
    out = tmp_path / "processed"
    lock = tmp_path / "data.lock"
    build_corpus(raw, out)

    digest = write_lock(out, lock)

    assert assert_frozen(out, lock) == digest
    assert json.loads(lock.read_text(encoding="utf-8"))["manifest_sha256"] == digest
    manifest_path = out / "splits_manifest.json"
    manifest_path.write_text(manifest_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ManifestMismatch, match="does not match"):
        assert_frozen(out, lock)


def test_missing_lock_is_refused(raw: Path, tmp_path: Path) -> None:
    out = tmp_path / "processed"
    build_corpus(raw, out)

    with pytest.raises(ManifestMismatch, match="fnd data freeze"):
        assert_frozen(out, tmp_path / "absent.lock")
