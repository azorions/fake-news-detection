"""FakeNewsNet (Shu et al., 2018) index files: headlines and source URLs from PolitiFact and GossipCop.

Article bodies are not redistributed, so every row is a headline example. The registrable domain of the
source URL is the publisher, which drives publisher-grouped splitting.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import tldextract

from fnd_ml.common.labels import RELIABLE, UNRELIABLE
from fnd_ml.data.schema import finalize

# Use the public suffix list bundled with tldextract; never fetch it from the network.
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=())

_FILES = {
    "politifact_fake.csv": UNRELIABLE,
    "politifact_real.csv": RELIABLE,
    "gossipcop_fake.csv": UNRELIABLE,
    "gossipcop_real.csv": RELIABLE,
}


def registrable_domain(url: str | None) -> str | None:
    if not url or not isinstance(url, str):
        return None
    domain = _EXTRACT(url.strip()).top_domain_under_public_suffix
    return domain.lower() or None


def load(directory: Path) -> pd.DataFrame:
    parts = []
    for filename, label in _FILES.items():
        part = pd.read_csv(directory / filename, dtype=str)
        part["label"] = label
        part["id"] = "fakenewsnet:" + filename.removesuffix(".csv") + ":" + part["id"]
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    urls = frame["news_url"].where(frame["news_url"].notna(), None)
    out = pd.DataFrame(
        {
            "id": frame["id"],
            "text": frame["title"].fillna("").str.strip(),
            "label": frame["label"],
            "url": urls,
            "publisher": urls.map(registrable_domain),
        }
    )
    return finalize(out, "fakenewsnet")
