"""ISOT Fake News (Ahmed et al., 2017): Fake.csv and True.csv inside one zip, 2016-2017.

Every true article is Reuters wire copy, so publisher artifacts must be scrubbed before training.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from fnd_ml.common.labels import RELIABLE, UNRELIABLE
from fnd_ml.data.schema import finalize


def load(directory: Path) -> pd.DataFrame:
    parts = []
    with zipfile.ZipFile(directory / "News_dataset.zip") as archive:
        for member in archive.namelist():
            name = Path(member).name.lower()
            if name not in ("fake.csv", "true.csv"):
                continue
            label = UNRELIABLE if name == "fake.csv" else RELIABLE
            with archive.open(member) as handle:
                part = pd.read_csv(handle, dtype=str)
            part["label"] = label
            part["id"] = f"isot:{name[:4]}:" + part.index.astype(str)
            parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    dates = pd.to_datetime(frame["date"].str.strip(), format="%B %d, %Y", errors="coerce")
    out = pd.DataFrame(
        {
            "id": frame["id"],
            "title": frame["title"].fillna("").str.strip().replace("", None),
            "text": frame["text"].fillna("").str.strip(),
            "label": frame["label"],
            "date": dates.dt.strftime("%Y-%m-%d").where(dates.notna(), None),
        }
    )
    return finalize(out, "isot")
