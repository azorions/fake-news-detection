"""LIAR statements (Wang, 2017) as distributed by LIAR-PLUS (Alhindi et al., 2018).

Six PolitiFact ratings become a binary label: true and mostly-true are reliable, false and pants-fire
are unreliable. half-true and barely-true are kept apart as a "mixed" evaluation slice
(``load_middle``). The speaker is the publisher, so one speaker's statements never span two splits.
The credit-history count columns include each statement's own rating and are never used.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from fnd_ml.common.labels import RELIABLE, UNRELIABLE
from fnd_ml.data.schema import finalize

BINARY_LABELS = {
    "true": RELIABLE,
    "mostly-true": RELIABLE,
    "false": UNRELIABLE,
    "pants-fire": UNRELIABLE,
}
MIDDLE_LABELS = frozenset({"half-true", "barely-true"})
_FILES = ("train2.tsv", "val2.tsv", "test2.tsv")
_COLUMNS = {
    1: "json_id",
    2: "rating",
    3: "statement",
    5: "speaker",
    14: "context",
    15: "justification",
}


def _read_all(directory: Path) -> pd.DataFrame:
    parts = []
    for filename in _FILES:
        part = pd.read_csv(
            directory / filename, sep="\t", header=None, quoting=csv.QUOTE_NONE, dtype=str
        )
        parts.append(part.reindex(columns=range(16)))
    frame = pd.concat(parts, ignore_index=True)[list(_COLUMNS)].rename(columns=_COLUMNS)
    # Rows broken by stray newlines in LIAR-PLUS justifications have no valid rating; drop them.
    frame = frame[frame["rating"].isin(BINARY_LABELS.keys() | MIDDLE_LABELS)]
    return frame.drop_duplicates("json_id")


def _to_schema(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": "liar:" + frame["json_id"].str.removesuffix(".json"),
            "text": frame["statement"].fillna("").str.strip(),
            "label": frame["rating"].map(BINARY_LABELS).fillna(-1).astype("int64"),
            "publisher": frame["speaker"].fillna("").str.strip().replace("", None),
        }
    )


def load(directory: Path) -> pd.DataFrame:
    frame = _read_all(directory)
    return finalize(_to_schema(frame[frame["rating"].isin(BINARY_LABELS)]), "liar")


def load_middle(directory: Path) -> pd.DataFrame:
    """The half-true and barely-true statements, with the original ``rating`` column.

    ``label`` is set to 1 only to satisfy the schema; these rows are for evaluation, never training.
    """
    frame = _read_all(directory)
    middle = frame[frame["rating"].isin(MIDDLE_LABELS)]
    out = finalize(_to_schema(middle).assign(label=UNRELIABLE), "liar")
    out["rating"] = middle["rating"].to_numpy()
    return out
