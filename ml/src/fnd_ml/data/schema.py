"""The common row schema every dataset loader returns."""

from __future__ import annotations

import pandas as pd

COLUMNS: list[str] = ["id", "dataset", "title", "text", "label", "publisher", "url", "date"]


def empty_frame(**extra: str) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in [*COLUMNS, *extra]})


def finalize(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Add missing columns, fix types and column order."""
    frame = frame.copy()
    frame["dataset"] = dataset
    for column in COLUMNS:
        if column not in frame:
            frame[column] = None
    frame["label"] = frame["label"].astype("int64")
    for column in ("title", "publisher", "url", "date"):
        frame[column] = frame[column].astype("object").where(frame[column].notna(), None)
    return frame[COLUMNS].reset_index(drop=True)


def validate_frame(frame: pd.DataFrame) -> None:
    """Raise ``ValueError`` if ``frame`` does not follow the schema."""
    if list(frame.columns[: len(COLUMNS)]) != COLUMNS:
        raise ValueError(f"columns must start with {COLUMNS}, got {list(frame.columns)}")
    for column in ("id", "dataset", "text", "label"):
        if frame[column].isna().any():
            raise ValueError(f"column {column!r} has missing values")
    bad_labels = set(frame["label"].unique()) - {0, 1}
    if bad_labels:
        raise ValueError(f"label must be 0 or 1, found {sorted(bad_labels)}")
    if not frame["id"].is_unique:
        raise ValueError("column 'id' has duplicates")
