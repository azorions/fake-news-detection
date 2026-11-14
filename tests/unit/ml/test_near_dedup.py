import pandas as pd

from fnd_ml.data.dedup import (
    count_cross_split_near_duplicates,
    drop_exact_duplicates,
    drop_near_duplicates,
)
from fnd_ml.data.minhash import clusters, signatures

BASE = " ".join(f"w{i}" for i in range(200))
# One word of 200 changed: 5 of 196 shingles differ, so the Jaccard similarity is 191/201 = 0.95.
EDITED = " ".join("x17" if i == 17 else f"w{i}" for i in range(200))
OTHER = " ".join(f"z{i}" for i in range(200))
HALF = " ".join(f"w{i}" if i % 2 else f"q{i}" for i in range(200))  # every second word replaced


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    base = {"dataset": "toy", "label": 0, "publisher": None, "url": None, "date": None}
    records = []
    for i, row in enumerate(rows):
        record = {**base, "id": f"toy:{i}", **row}
        record.setdefault("model_text", record["text"])
        record.setdefault("text_hash", f"h{hash(record['text'])}")
        records.append(record)
    return pd.DataFrame(records)


# ---------------------------------------------------------------- exact duplicates keep metadata


def test_exact_duplicate_keeps_metadata_from_the_dropped_copy() -> None:
    frame = _frame(
        [
            {"dataset": "welfake", "text": BASE},
            {"dataset": "isot", "text": BASE, "date": "2017-03-01", "publisher": "reuters"},
            {"dataset": "welfake", "text": OTHER},
        ]
    )

    kept, stats = drop_exact_duplicates(frame)

    assert len(kept) == 2
    assert kept.loc[0, "dataset"] == "welfake"
    assert str(kept.loc[0, "date"])[:10] == "2017-03-01"
    assert kept.loc[0, "publisher"] == "reuters"
    assert kept.loc[0, "also_in"] == "isot"
    assert kept.loc[1, "also_in"] == ""
    assert kept.loc[1, "date"] is None or pd.isna(kept.loc[1, "date"])
    assert stats == {"input": 3, "duplicates_removed": 1, "conflicting_removed": 0, "output": 2}


def test_the_kept_rows_own_metadata_wins() -> None:
    frame = _frame(
        [
            {"dataset": "fakenewsnet", "text": BASE, "publisher": "first.example"},
            {"dataset": "welfake", "text": BASE, "publisher": "second.example"},
            {"dataset": "fakenewsnet", "text": BASE},
        ]
    )

    kept, _ = drop_exact_duplicates(frame)

    assert kept.loc[0, "publisher"] == "first.example"
    assert kept.loc[0, "also_in"] == "welfake"  # the row's own dataset is not listed


def test_exact_duplicates_with_both_labels_are_still_dropped() -> None:
    frame = _frame([{"text": BASE, "label": 0}, {"text": BASE, "label": 1}, {"text": OTHER}])

    kept, stats = drop_exact_duplicates(frame)

    assert kept["text"].tolist() == [OTHER]
    assert stats["conflicting_removed"] == 2


# ---------------------------------------------------------------- MinHash


def test_near_duplicates_form_one_cluster() -> None:
    sig, valid = signatures([BASE, EDITED, OTHER])
    ids = clusters(sig, valid)

    assert valid.all()
    assert ids[0] == ids[1] != ids[2]
    assert ids.tolist() == [0, 0, 2]  # a cluster is named after its first row


def test_half_rewritten_text_stays_apart() -> None:
    sig, valid = signatures([BASE, HALF])

    assert len(set(clusters(sig, valid).tolist())) == 2


def test_short_texts_are_never_merged() -> None:
    sig, valid = signatures(["breaking news today", "breaking news today"])

    assert not valid.any()
    assert len(set(clusters(sig, valid).tolist())) == 2


def test_signature_is_the_same_in_any_position() -> None:
    alone, _ = signatures([BASE])
    second, _ = signatures(["other words " * 9, BASE])

    assert alone.shape == (1, 128)
    assert (alone[0] == second[1]).all()


def test_case_and_punctuation_do_not_matter() -> None:
    sig, valid = signatures([BASE, BASE.upper().replace(" ", ", ")])

    assert (sig[0] == sig[1]).all()
    assert valid.all()


def test_a_chain_of_edits_is_one_cluster() -> None:
    step_two = EDITED.replace("w150", "x150")
    sig, valid = signatures([OTHER, BASE, EDITED, step_two])

    assert clusters(sig, valid).tolist() == [0, 1, 1, 1]


# ---------------------------------------------------------------- near-duplicate removal


def test_near_duplicate_is_removed_and_its_metadata_kept() -> None:
    frame = _frame(
        [
            {"dataset": "welfake", "text": BASE},
            {"dataset": "fakenewsnet", "text": EDITED, "publisher": "site.example", "url": "u"},
            {"dataset": "welfake", "text": OTHER},
            {"dataset": "liar", "text": "too short here"},
            {"dataset": "liar", "text": "too short here"},
        ]
    )

    kept, stats = drop_near_duplicates(frame)

    assert kept["text"].tolist() == [BASE, OTHER, "too short here", "too short here"]
    assert kept.loc[0, "publisher"] == "site.example"
    assert kept.loc[0, "also_in"] == "fakenewsnet"
    assert stats == {
        "input": 5,
        "clusters_with_duplicates": 1,
        "near_duplicates_removed": 1,
        "conflicting_removed": 0,
        "output": 4,
    }


def test_also_in_from_the_exact_step_is_carried_along() -> None:
    frame = _frame(
        [
            {"dataset": "welfake", "text": BASE, "also_in": "isot"},
            {"dataset": "fakenewsnet", "text": EDITED, "also_in": ""},
        ]
    )

    kept, _ = drop_near_duplicates(frame)

    assert kept.loc[0, "also_in"] == "fakenewsnet,isot"


def test_cluster_with_both_labels_is_dropped_whole() -> None:
    frame = _frame(
        [{"text": BASE, "label": 0}, {"text": EDITED, "label": 1}, {"text": OTHER, "label": 1}]
    )

    kept, stats = drop_near_duplicates(frame)

    assert kept["text"].tolist() == [OTHER]
    assert stats["conflicting_removed"] == 2
    assert stats["near_duplicates_removed"] == 0


def test_nothing_crosses_splits_after_removal() -> None:
    frame = _frame(
        [
            {"text": BASE, "split": "train"},
            {"text": EDITED, "split": "test"},
            {"text": OTHER, "split": "test"},
        ]
    )
    assert count_cross_split_near_duplicates(frame) == 1

    kept, _ = drop_near_duplicates(frame)

    assert count_cross_split_near_duplicates(kept) == 0
