from pathlib import Path

import numpy as np
import pandas as pd

from fnd_ml.data.leak_probe import (
    count_unnamed,
    label_probe,
    load_identifiers,
    load_probe_sample,
    origin_probe,
)

REGISTRY_FILE = Path("data/registry/source_identifiers.txt")


def _texts(rng: np.random.Generator, prefix: str, rows: int) -> list[str]:
    return [" ".join(f"{prefix}{n}" for n in rng.integers(10, 60, size=40)) for _ in range(rows)]


def _corpus(prefixes: dict[str, str], rows: int = 150, seed: int = 0) -> pd.DataFrame:
    """Each dataset draws its words from the vocabulary named by its prefix."""
    rng = np.random.default_rng(seed)
    parts = []
    for dataset, prefix in prefixes.items():
        for split in ("train", "val"):
            parts.append(
                pd.DataFrame(
                    {
                        "dataset": dataset,
                        "split": split,
                        "label": rng.integers(0, 2, size=rows),
                        "model_text": _texts(rng, prefix, rows),
                    }
                )
            )
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------- identifiers


def test_identifiers_are_lowercased_and_comments_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "ids.txt"
    path.write_text("# wires\nReuters\n\n  York Times  # two words\nrt\n", encoding="utf-8")

    assert load_identifiers(path) == frozenset({"reuters", "york times", "rt"})


def test_the_shipped_list_names_the_sources_found_in_m1() -> None:
    identifiers = load_identifiers(REGISTRY_FILE)

    assert {
        "reuters",
        "breitbart",
        "sputnik",
        "infowars",
        "politifact",
        "york times",
    } <= identifiers
    # Ordinary English words would flag honest features, so publishers like people.com are left out.
    assert not {"people", "time", "today", "variety"} & identifiers


# ---------------------------------------------------------------- label probe


def test_label_probe_flags_identifier_tokens() -> None:
    top = {"unreliable": ["shocking", "breitbart news", "via"], "reliable": ["said on", "reuters"]}

    assert label_probe(top, frozenset({"breitbart", "reuters"})) == ["breitbart news", "reuters"]


def test_label_probe_matches_whole_tokens_only() -> None:
    assert label_probe({"unreliable": ["report", "part of", "rts"]}, frozenset({"rt"})) == []


def test_two_word_identifier() -> None:
    top = {"reliable": ["york times", "new york", "times"]}

    assert label_probe(top, frozenset({"york times"})) == ["york times"]


def test_unnamed_buckets_are_counted_not_flagged() -> None:
    top = {"unreliable": ["<bucket 12>", "wow"], "reliable": ["<bucket 7>", "<bucket 9>"]}

    assert label_probe(top, frozenset({"bucket", "12"})) == []
    assert count_unnamed(top) == 3


# ---------------------------------------------------------------- origin probe


def test_origin_probe_separates_distinct_vocabularies() -> None:
    result = origin_probe(_corpus({"alpha": "aa", "beta": "bb"}))

    assert result["accuracy"] > 0.95
    assert result["majority_baseline"] == 0.5
    assert result["rows"] == {"train": 300, "val": 300}


def test_origin_probe_is_near_chance_for_one_vocabulary() -> None:
    result = origin_probe(_corpus({"alpha": "cc", "beta": "cc"}))

    assert result["accuracy"] < result["majority_baseline"] + 0.15


def test_origin_probe_reports_each_label_on_its_own() -> None:
    result = origin_probe(_corpus({"alpha": "aa", "beta": "bb"}))

    assert set(result["per_label"]) == {"0", "1"}
    assert result["per_label"]["0"]["accuracy"] > 0.95
    assert result["per_label"]["1"]["macro_f1"] > 0.95


def test_origin_probe_caps_rows_per_dataset() -> None:
    result = origin_probe(_corpus({"alpha": "aa", "beta": "bb"}), max_rows_per_dataset=40)

    assert result["rows"] == {"train": 80, "val": 80}


def test_a_label_with_one_dataset_has_no_number() -> None:
    frame = _corpus({"alpha": "aa", "beta": "bb"})
    frame.loc[frame["dataset"] == "beta", "label"] = 1  # label 0 now exists in one dataset only

    result = origin_probe(frame)

    assert result["per_label"]["0"] is None
    assert result["per_label"]["1"] is not None


# ---------------------------------------------------------------- sampling from disk


def test_probe_sample_reads_a_capped_part_of_the_split_files(tmp_path: Path) -> None:
    frame = _corpus({"alpha": "aa", "beta": "bb"}, rows=60)
    frame["title"] = "unused column"
    (tmp_path / "splits").mkdir()
    for split in ("train", "val"):
        frame[frame["split"] == split].to_parquet(tmp_path / "splits" / f"{split}.parquet")

    sample = load_probe_sample(tmp_path, max_rows_per_dataset=25, seed=13)

    assert sorted(sample.columns) == ["dataset", "label", "model_text", "split"]
    assert sample.groupby(["split", "dataset"]).size().tolist() == [25, 25, 25, 25]
    again = load_probe_sample(tmp_path, max_rows_per_dataset=25, seed=13)
    assert sample["model_text"].tolist() == again["model_text"].tolist()
