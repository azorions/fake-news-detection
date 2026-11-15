"""BuzzFeed-Webis Fake News Corpus 2016 (Potthast et al., 2018): 1,627 articles from nine
publishers, each rated by BuzzFeed journalists. Evaluation only.

"mostly false" and "mixture of true and false" count as unreliable, "mostly true" as reliable.
Articles with "no factual content" (opinion, satire) have nothing to judge and are left out.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd

from fnd_ml.data.schema import finalize

_ARCHIVE = "articles.zip"
_LABELS = {"mostly true": 0, "mixture of true and false": 1, "mostly false": 1}
_SKIPPED = {"no factual content"}


def _field(article: ElementTree.Element, tag: str) -> str:
    return (article.findtext(tag) or "").strip()


def load(directory: Path) -> pd.DataFrame:
    rows = []
    with zipfile.ZipFile(directory / _ARCHIVE) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith(".xml"):
                continue
            # The archive's sha256 is pinned in checksums.json, so this is known content.
            article = ElementTree.fromstring(archive.read(name))
            veracity = _field(article, "veracity")
            text = _field(article, "mainText")
            if veracity in _SKIPPED or not text:
                continue
            if veracity not in _LABELS:
                raise ValueError(f"buzzfeed_webis/{name}: unknown veracity {veracity!r}")
            rows.append(
                {
                    "id": f"buzzfeed_webis:{Path(name).stem}",
                    "title": _field(article, "title") or None,
                    "text": text,
                    "label": _LABELS[veracity],
                    "publisher": _field(article, "portal") or None,
                    "url": _field(article, "uri") or None,
                }
            )
    columns = ["id", "title", "text", "label", "publisher", "url"]
    return finalize(pd.DataFrame(rows, columns=columns), "buzzfeed_webis")
