"""A processed-data folder (split files, manifest, lock) written from a small frame, for training tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fnd_ml.data.build import MANIFEST_NAME, write_lock
from tests.fixtures.synthetic import make_corpus

BOUNDS = {"train": (0, 240), "val": (240, 290), "calib": (290, 350), "test": (350, 400)}


def write_processed(root: Path, frame: pd.DataFrame, *, freeze: bool = True) -> tuple[Path, Path]:
    """Write ``frame`` as the four splits under ``root``; return the processed folder and its lock."""
    processed, lock = root / "processed", root / "data.lock"
    (processed / "splits").mkdir(parents=True)
    for split, (start, end) in BOUNDS.items():
        frame.iloc[start:end].to_parquet(processed / "splits" / f"{split}.parquet", index=False)
    manifest = {
        "seed": 13,
        "splits_version": 2,
        "rows": {split: end - start for split, (start, end) in BOUNDS.items()},
    }
    (processed / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    if freeze:
        write_lock(processed, lock)
    return processed, lock


def two_dataset_corpus(n: int = 400, seed: int = 2) -> pd.DataFrame:
    """The synthetic corpus with its rows attributed to two datasets, both holding both labels."""
    frame = make_corpus(n, seed=seed)
    return frame.assign(dataset=np.where(np.arange(n) % 4 < 2, "alpha", "beta"))


def stylistic_corpus(n: int, seed: int) -> pd.DataFrame:
    """Both labels use the same words; label 1 is written in capitals and ends with "!!!".

    A model that only sees lower-cased words cannot tell the labels apart.
    """
    neutral = make_corpus(2 * n, seed=seed)
    neutral = neutral[neutral["label"] == 0].reset_index(drop=True)
    labels = np.arange(n) % 2
    texts = [
        text.upper().rstrip(".") + "!!!" if label else text
        for text, label in zip(neutral["text"], labels, strict=True)
    ]
    return neutral.assign(text=texts, model_text=texts, label=labels)
