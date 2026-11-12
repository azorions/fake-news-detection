"""Exact-duplicate removal across all datasets (near-duplicates follow in M2)."""

from __future__ import annotations

import pandas as pd


def drop_exact_duplicates(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep one row per ``text_hash``.

    A hash that occurs with both labels is ambiguous ground truth, so every row with it is removed.
    Otherwise the first row (in load order) is kept.
    """
    labels_per_hash = frame.groupby("text_hash")["label"].nunique()
    conflicting = frame["text_hash"].isin(labels_per_hash.index[labels_per_hash > 1])
    consistent = frame[~conflicting]
    kept = consistent.drop_duplicates("text_hash", keep="first").reset_index(drop=True)
    stats = {
        "input": len(frame),
        "duplicates_removed": len(consistent) - len(kept),
        "conflicting_removed": int(conflicting.sum()),
        "output": len(kept),
    }
    return kept, stats
