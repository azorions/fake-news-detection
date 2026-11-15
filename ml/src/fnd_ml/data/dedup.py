"""Duplicate removal across all datasets: exact (same text hash) and near (MinHash).

One row is kept per group of duplicates. It keeps its own dataset and text, takes the first
``date``, ``publisher`` and ``url`` that any copy has (the datasets overlap, and the copy that is
kept is often the one without metadata), and lists the other datasets in ``also_in``.
"""

from __future__ import annotations

import pandas as pd

from fnd_ml.data.minhash import clusters, signatures

_METADATA = ("date", "publisher", "url")
_KEY = "_duplicate_key"


def _collapse(frame: pd.DataFrame, key: pd.Series) -> tuple[pd.DataFrame, int, int]:
    """Keep the first row of every ``key`` group; drop groups that contain both labels.

    Returns the kept rows, the number of duplicate rows removed and the number of rows removed
    because their group's labels conflict.
    """
    frame = frame.reset_index(drop=True)
    frame[_KEY] = key.to_numpy()
    if "also_in" not in frame:
        frame["also_in"] = ""

    labels_per_key = frame.groupby(_KEY)["label"].nunique()
    conflicting = frame[_KEY].isin(labels_per_key.index[labels_per_key > 1])
    consistent = frame[~conflicting]
    kept = consistent.drop_duplicates(_KEY, keep="first").copy()

    repeated = consistent[consistent[_KEY].duplicated(keep=False)]
    if len(repeated):
        first_values = repeated.groupby(_KEY, sort=False)[list(_METADATA)].first()
        for column in _METADATA:
            inherited = kept[_KEY].map(first_values[column])
            kept[column] = kept[column].where(kept[column].notna(), inherited)
            kept[column] = kept[column].astype("object").where(kept[column].notna(), None)

        carried = repeated.loc[repeated["also_in"] != "", [_KEY, "also_in"]]
        carried = carried.assign(dataset=carried["also_in"].str.split(",")).explode("dataset")
        seen = pd.concat([repeated[[_KEY, "dataset"]], carried[[_KEY, "dataset"]]])
        datasets_per_key = seen.drop_duplicates().groupby(_KEY)["dataset"].agg(sorted)
        in_group = kept[_KEY].isin(datasets_per_key.index)
        kept.loc[in_group, "also_in"] = [
            ",".join(name for name in datasets_per_key[group] if name != own)
            for group, own in zip(
                kept.loc[in_group, _KEY], kept.loc[in_group, "dataset"], strict=True
            )
        ]

    removed = len(consistent) - len(kept)
    return kept.drop(columns=_KEY).reset_index(drop=True), removed, int(conflicting.sum())


def drop_exact_duplicates(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep one row per ``text_hash``.

    A hash that occurs with both labels is ambiguous ground truth, so every row with it is removed.
    Otherwise the first row (in load order) is kept, with the metadata of its copies.
    """
    kept, removed, conflicting = _collapse(frame, frame["text_hash"])
    stats = {
        "input": len(frame),
        "duplicates_removed": removed,
        "conflicting_removed": conflicting,
        "output": len(kept),
    }
    return kept, stats


def _near_duplicate_clusters(frame: pd.DataFrame, threshold: float, seed: int) -> pd.Series:
    sig, valid = signatures(frame["model_text"], seed=seed)
    return pd.Series(clusters(sig, valid, threshold=threshold))


def drop_near_duplicates(
    frame: pd.DataFrame, *, threshold: float = 0.8, seed: int = 13
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep one row per cluster of near-duplicate ``model_text`` (5-word shingles, Jaccard at
    least ``threshold``). Clusters that contain both labels are removed whole. Texts shorter than
    five words are never near-duplicates of anything."""
    cluster = _near_duplicate_clusters(frame, threshold, seed)
    kept, removed, conflicting = _collapse(frame, cluster)
    stats = {
        "input": len(frame),
        "clusters_with_duplicates": int((cluster.value_counts() > 1).sum()),
        "near_duplicates_removed": removed,
        "conflicting_removed": conflicting,
        "output": len(kept),
    }
    return kept, stats


def count_cross_split_near_duplicates(
    frame: pd.DataFrame, *, threshold: float = 0.8, seed: int = 13
) -> int:
    """How many near-duplicate clusters have rows in more than one split (the hygiene gate)."""
    cluster = _near_duplicate_clusters(frame, threshold, seed)
    splits_per_cluster = frame["split"].groupby(cluster.to_numpy()).nunique()
    return int((splits_per_cluster > 1).sum())
