"""News headlines for sarcasm detection (Misra and Arora, 2019): The Onion against HuffPost.

Used only for the satire scorer. In this dataset ``label`` means satire (1) or news (0); it does
not mean unreliable.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from fnd_ml.data.schema import finalize

_FILE = "Sarcasm_Headlines_Dataset.json"
_PUBLISHER = re.compile(r"(theonion|huffingtonpost)\.com")


def _publisher(link: str) -> str | None:
    match = _PUBLISHER.search(link)
    return match.group(0) if match else None


def load(directory: Path) -> pd.DataFrame:
    frame = pd.read_json(directory / _FILE, lines=True, dtype=False)
    out = pd.DataFrame(
        {
            "id": "satire_headlines:" + pd.Series(range(len(frame)), dtype="int64").astype(str),
            "text": frame["headline"].astype(str).str.strip(),
            "label": frame["is_sarcastic"].astype("int64"),
            "publisher": frame["article_link"].astype(str).map(_publisher),
            "url": frame["article_link"].astype(str),
        }
    )
    return finalize(out[out["text"] != ""], "satire_headlines")
