"""WELFake (Verma et al., 2021): ~72k articles merged from four corpora.

The published description of WELFake's labels has been disputed. The data itself shows that every row
with a "(Reuters) -" dateline has label 0, so label 0 is reliable, which already matches this project's
convention. ``fnd data build`` re-checks this with ``wire_label_share``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fnd_ml.data.schema import finalize


def load(directory: Path) -> pd.DataFrame:
    frame = pd.read_csv(directory / "WELFake_Dataset.csv", dtype={"title": str, "text": str})
    frame = frame.rename(columns={frame.columns[0]: "row"})
    title = frame["title"].fillna("").str.strip()
    text = frame["text"].fillna("").str.strip()
    body_missing = text == ""
    out = pd.DataFrame(
        {
            "id": "welfake:" + frame["row"].astype(str),
            # Rows without a body become headline examples: the title moves into the text.
            "title": title.where(~body_missing & (title != ""), None),
            "text": text.where(~body_missing, title),
            "label": frame["label"],
        }
    )
    return finalize(out, "welfake")
