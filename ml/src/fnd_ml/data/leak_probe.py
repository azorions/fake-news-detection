"""Leak probes: is the classifier reading who published a text instead of how it is written?

* Label probe: no name of a source may be among the baseline's strongest features.
* Origin probe: how well can a simple model tell which dataset a scrubbed text came from? The
  datasets differ in label balance, so whatever makes the dataset recognizable is a shortcut to the
  label. The number is reported, not gated: it measures what scrubbing left behind.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from fnd_ml.common.tokens import analyze

_UNNAMED = "<bucket "
_PROBE_SPLITS = ("train", "val")
_COLUMNS = ["dataset", "label", "model_text"]


def load_identifiers(path: Path) -> frozenset[str]:
    """Source names, one per line, lower-cased; ``#`` starts a comment."""
    lines = path.read_text(encoding="utf-8").splitlines()
    terms = (" ".join(line.split("#", 1)[0].lower().split()) for line in lines)
    return frozenset(term for term in terms if term)


def label_probe(top_features: dict[str, list[str]], identifiers: frozenset[str]) -> list[str]:
    """The features that name a source: one of their words is an identifier, or the whole feature
    is a two-word identifier. Unnamed hash buckets cannot be checked (see ``count_unnamed``)."""
    flagged = []
    for features in top_features.values():
        for feature in features:
            if feature.startswith(_UNNAMED):
                continue
            if feature in identifiers or any(word in identifiers for word in feature.split()):
                flagged.append(feature)
    return flagged


def count_unnamed(top_features: dict[str, list[str]]) -> int:
    """How many top features are hash buckets without a name, which the label probe cannot check."""
    return sum(
        feature.startswith(_UNNAMED) for features in top_features.values() for feature in features
    )


def _cap(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    parts = [
        group.sample(n=min(len(group), max_rows), random_state=seed)
        for _, group in frame.groupby("dataset", sort=True)
    ]
    return pd.concat(parts) if parts else frame


def _probe(train: pd.DataFrame, val: pd.DataFrame, seed: int) -> dict[str, Any] | None:
    if train["dataset"].nunique() < 2 or val["dataset"].nunique() < 2:
        return None
    vectorizer = HashingVectorizer(analyzer=analyze, n_features=2**18, alternate_sign=False)
    model = LogisticRegression(max_iter=300, random_state=seed)
    model.fit(vectorizer.transform(train["model_text"]), train["dataset"])
    predicted = model.predict(vectorizer.transform(val["model_text"]))
    return {
        "accuracy": round(float(accuracy_score(val["dataset"], predicted)), 4),
        "majority_baseline": round(float(val["dataset"].value_counts(normalize=True).iloc[0]), 4),
        "macro_f1": round(float(f1_score(val["dataset"], predicted, average="macro")), 4),
        "rows": {"train": len(train), "val": len(val)},
    }


def origin_probe(
    frame: pd.DataFrame, *, seed: int = 13, max_rows_per_dataset: int = 8000
) -> dict[str, Any]:
    """Predict ``dataset`` from ``model_text``: fit on the train split, score on val.

    ``per_label`` repeats the probe inside each label, so that the label itself cannot be what
    gives the dataset away; a label found in fewer than two datasets has ``None``.
    """
    train = _cap(frame[frame["split"] == "train"], max_rows_per_dataset, seed)
    val = _cap(frame[frame["split"] == "val"], max_rows_per_dataset, seed)
    result = _probe(train, val, seed)
    if result is None:
        raise ValueError("the origin probe needs at least two datasets in train and in val")
    result["per_label"] = {
        str(label): _probe(train[train["label"] == label], val[val["label"] == label], seed)
        for label in (0, 1)
    }
    return result


def load_probe_sample(
    processed_dir: Path, *, max_rows_per_dataset: int = 8000, seed: int = 13
) -> pd.DataFrame:
    """A capped random sample of the train and val split files, read without loading them whole."""
    rng = np.random.default_rng(seed)
    parts = []
    for split in _PROBE_SPLITS:
        path = processed_dir / "splits" / f"{split}.parquet"
        datasets = pq.read_table(path, columns=["dataset"]).column("dataset").to_pandas()
        wanted = np.zeros(len(datasets), dtype=bool)
        for _, positions in datasets.groupby(datasets, sort=True).indices.items():
            size = min(len(positions), max_rows_per_dataset)
            wanted[rng.choice(positions, size=size, replace=False)] = True

        offset = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048, columns=_COLUMNS):
            chunk = batch.to_pandas()
            parts.append(chunk[wanted[offset : offset + len(chunk)]].assign(split=split))
            offset += len(chunk)
    return pd.concat(parts, ignore_index=True)
