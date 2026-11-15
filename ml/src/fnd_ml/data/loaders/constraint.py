"""Constraint@AAAI-2021 COVID-19 fake news (Patwa et al., 2021): social media posts from 2020.

Evaluation only. Its three published parts (train, val, test) are all used as one external test
set, because nothing in this project is trained or tuned on it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fnd_ml.data.schema import finalize

_FILES = {
    "train": "Constraint_Train.csv",
    "val": "Constraint_Val.csv",
    "test": "english_test_with_labels.csv",
}
_LABELS = {"fake": 1, "real": 0}


def load(directory: Path) -> pd.DataFrame:
    parts = []
    for part, filename in _FILES.items():
        frame = pd.read_csv(directory / filename, dtype=str, keep_default_na=False)
        unknown = sorted(set(frame["label"]) - set(_LABELS))
        if unknown:
            raise ValueError(f"constraint/{filename}: unknown label(s) {unknown}")
        parts.append(
            pd.DataFrame(
                {
                    "id": f"constraint:{part}:" + frame["id"],
                    "text": frame["tweet"].str.strip(),
                    "label": frame["label"].map(_LABELS),
                }
            )
        )
    out = pd.concat(parts, ignore_index=True)
    return finalize(out[out["text"] != ""], "constraint")
