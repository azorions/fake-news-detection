"""Article text and metadata from a page's HTML."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import datetime

import trafilatura

from fnd_api.ingest.metadata import DateSource, parse_html, parse_metadata
from fnd_api.ingest.paywall import looks_paywalled

_THIN_TEXT_WORDS = 40
_NOT_CONTENT = ("script", "style", "noscript", "nav", "header", "footer", "aside", "form")

# trafilatura reports unparseable pages at ERROR level; here that is an ordinary outcome.
logging.getLogger("trafilatura").setLevel(logging.CRITICAL)


@dataclass(frozen=True)
class ExtractedArticle:
    title: str | None
    text: str
    authors: list[str]
    site_name: str | None
    published_at: datetime | None
    modified_at: datetime | None
    date_source: DateSource
    canonical_url: str | None
    paywall_hint: bool


def _paragraph_text(html: str) -> str:
    """Plain paragraphs of the page, for pages too small or unusual for the main extractor."""
    tree = parse_html(html)
    if tree is None:
        return ""
    for element in tree.xpath("|".join(f"//{tag}" for tag in _NOT_CONTENT)):
        element.drop_tree()
    paragraphs = tree.xpath("//article//p") or tree.xpath("//main//p") or tree.xpath("//p")
    texts = (" ".join(paragraph.text_content().split()) for paragraph in paragraphs)
    return "\n\n".join(text for text in texts if text)


def _main_text(html: str, url: str) -> str:
    try:
        extracted = trafilatura.extract(
            html, url=url, include_comments=False, include_tables=False, favor_precision=True
        )
    except Exception:  # hostile markup must not take the request down
        extracted = None
    text = (extracted or "").strip()
    if len(text.split()) < _THIN_TEXT_WORDS:
        fallback = _paragraph_text(html)
        if len(fallback.split()) > len(text.split()):
            text = fallback
    return text


def extract_article(html: str, url: str) -> ExtractedArticle:
    """Never raises: a page without usable content gives an article with little or no text."""
    meta = parse_metadata(html, url)
    article = ExtractedArticle(
        title=meta.title,
        text=_main_text(html, url),
        authors=meta.authors,
        site_name=meta.site_name,
        published_at=meta.published_at,
        modified_at=meta.modified_at,
        date_source=meta.date_source,
        canonical_url=meta.canonical_url,
        paywall_hint=False,
    )
    return replace(article, paywall_hint=looks_paywalled(article, html))
