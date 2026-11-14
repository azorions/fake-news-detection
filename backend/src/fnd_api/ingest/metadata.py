"""Page metadata from JSON-LD, meta tags and the URL: title, authors, site, dates, canonical URL."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import urljoin, urlsplit

import lxml.html
from lxml import etree

DateSource = Literal["jsonld", "meta", "url", "text", "none"]

_PUBLISHED_META = (
    "article:published_time",
    "og:article:published_time",
    "parsely-pub-date",
    "dc.date",
    "dc.date.issued",
    "date",
    "pubdate",
    "publish-date",
)
_MODIFIED_META = ("article:modified_time", "og:updated_time", "last-modified")
_AUTHOR_META = ("author", "parsely-author", "dc.creator", "article:author")
_URL_DATE = re.compile(r"/((?:19|20)\d{2})/(\d{1,2})/(\d{1,2})(?=[/-]|$)")
_TITLE_SEPARATORS = (" | ", " - ", " – ", " — ", " :: ", " · ")
_MAX_AUTHORS = 10


@dataclass(frozen=True)
class PageMetadata:
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    site_name: str | None = None
    published_at: datetime | None = None
    modified_at: datetime | None = None
    date_source: DateSource = "none"
    canonical_url: str | None = None


def parse_html(html: str) -> lxml.html.HtmlElement | None:
    """Parse leniently; ``None`` when there is nothing to parse."""
    if not html.strip():
        return None
    try:
        parser = lxml.html.HTMLParser(encoding="utf-8")
        return lxml.html.document_fromstring(html.encode("utf-8", "replace"), parser=parser)
    except (etree.ParserError, etree.XMLSyntaxError, ValueError):
        return None


def _clean(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    return text or None


def _parse_date(value: Any) -> datetime | None:
    text = _clean(value)
    if text is None:
        return None
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def _date_from_url(url: str) -> datetime | None:
    found = _URL_DATE.search(urlsplit(url).path)
    if not found:
        return None
    try:
        return datetime(int(found[1]), int(found[2]), int(found[3]), tzinfo=UTC)
    except ValueError:
        return None


def _json_ld_nodes(tree: lxml.html.HtmlElement) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for script in tree.xpath('//script[@type="application/ld+json"]'):
        try:
            document = json.loads(script.text_content(), strict=False)
        except (ValueError, RecursionError):
            continue
        pending = document if isinstance(document, list) else [document]
        for item in pending:
            if not isinstance(item, dict):
                continue
            nodes.append(item)
            graph = item.get("@graph")
            if isinstance(graph, list):
                nodes.extend(node for node in graph if isinstance(node, dict))
    return nodes


def _types(node: dict[str, Any]) -> list[str]:
    kind = node.get("@type")
    kinds = kind if isinstance(kind, list) else [kind]
    return [k for k in kinds if isinstance(k, str)]


def _is_article(node: dict[str, Any]) -> bool:
    return any(kind.endswith(("Article", "Posting")) or kind == "Report" for kind in _types(node))


def _names(value: Any) -> list[str]:
    """Names from a JSON-LD ``author``/``publisher`` value: a string, an object, or a list of them."""
    items = value if isinstance(value, list) else [value]
    names: list[str] = []
    for item in items:
        name = _clean(item.get("name")) if isinstance(item, dict) else _clean(item)
        if name and not name.lower().startswith(("http://", "https://")) and name not in names:
            names.append(name)
    return names[:_MAX_AUTHORS]


def _meta_tags(tree: lxml.html.HtmlElement) -> dict[str, str]:
    tags: dict[str, str] = {}
    for element in tree.xpath("//meta[@content]"):
        key = element.get("property") or element.get("name") or element.get("itemprop")
        content = _clean(element.get("content"))
        if key and content:
            tags.setdefault(key.strip().lower(), content)
    return tags


def _first(tags: dict[str, str], keys: tuple[str, ...]) -> str | None:
    return next((tags[key] for key in keys if key in tags), None)


def _strip_site_suffix(title: str, site_name: str | None) -> str:
    if site_name:
        for separator in _TITLE_SEPARATORS:
            head, found, tail = title.rpartition(separator)
            if found and head and tail.strip().lower() == site_name.lower():
                return head.strip()
    return title


def parse_metadata(html: str, url: str) -> PageMetadata:
    tree = parse_html(html)
    if tree is None:
        return PageMetadata()

    nodes = _json_ld_nodes(tree)
    article = next((node for node in nodes if _is_article(node)), {})
    tags = _meta_tags(tree)

    site_names = _names(article.get("publisher")) or [
        name
        for node in nodes
        if "WebSite" in _types(node) or "Organization" in _types(node)
        for name in _names(node)
    ]
    site_name = (site_names[0] if site_names else None) or tags.get("og:site_name")

    page_title = _clean(next(iter(tree.xpath("//title/text()")), None))
    heading = next(iter(tree.xpath("//h1")), None)
    title = (
        _clean(article.get("headline"))
        or tags.get("og:title")
        or (_strip_site_suffix(page_title, site_name) if page_title else None)
        or (_clean(heading.text_content()) if heading is not None else None)
    )

    authors = _names(article.get("author"))
    if not authors:
        meta_author = _first(tags, _AUTHOR_META)
        authors = _names(re.split(r",| and ", meta_author)) if meta_author else []

    published_at, modified_at, date_source = _parse_date(article.get("datePublished")), None, "none"
    if published_at is not None:
        date_source = "jsonld"
        modified_at = _parse_date(article.get("dateModified"))
    else:
        published_at = _parse_date(_first(tags, _PUBLISHED_META))
        if published_at is not None:
            date_source = "meta"
            modified_at = _parse_date(_first(tags, _MODIFIED_META))
        else:
            published_at = _date_from_url(url)
            date_source = "url" if published_at is not None else "none"

    canonical = _clean(next(iter(tree.xpath('//link[@rel="canonical"]/@href')), None))
    canonical = canonical or tags.get("og:url")
    canonical_url = urljoin(url, canonical) if canonical else None
    if canonical_url and urlsplit(canonical_url).scheme not in ("http", "https"):
        canonical_url = None

    return PageMetadata(
        title=title,
        authors=authors,
        site_name=site_name,
        published_at=published_at,
        modified_at=modified_at,
        date_source=date_source,  # type: ignore[arg-type]
        canonical_url=canonical_url,
    )
