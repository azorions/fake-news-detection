"""Detect pages that show only part of an article."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fnd_api.ingest.extractor import ExtractedArticle

SHORT_TEXT_WORDS = 120
_NOT_FREE = re.compile(r'"isAccessibleForFree"\s*:\s*"?false"?', re.IGNORECASE)
_PHRASES = (
    "subscribe to continue reading",
    "subscribe to read",
    "subscribe to keep reading",
    "subscribe now to read",
    "to continue reading",
    "already a subscriber",
    "subscribers only",
    "for subscribers only",
    "this article is for subscribers",
    "sign in to continue",
    "log in to continue",
    "log in to read",
    "create a free account to",
    "you have reached your article limit",
    "you've reached your limit of free articles",
)


def looks_paywalled(article: ExtractedArticle, html: str) -> bool:
    """True when the page declares itself not free (JSON-LD ``isAccessibleForFree: false``), or
    the extracted text is short and the page carries a paywall phrase."""
    if _NOT_FREE.search(html):
        return True
    if len(article.text.split()) >= SHORT_TEXT_WORDS:
        return False
    lowered = html.lower()
    return any(phrase in lowered for phrase in _PHRASES)
