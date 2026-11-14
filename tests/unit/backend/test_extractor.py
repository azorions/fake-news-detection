from datetime import UTC, datetime
from pathlib import Path

import pytest

from fnd_api.ingest.extractor import ExtractedArticle, extract_article
from fnd_api.ingest.metadata import parse_metadata
from fnd_api.ingest.paywall import looks_paywalled

FIXTURES = Path(__file__).parents[3] / "data" / "fixtures" / "html"


def load(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def words(text: str) -> int:
    return len(text.split())


# ---------------------------------------------------------------- news page with JSON-LD


@pytest.fixture(scope="module")
def news() -> ExtractedArticle:
    return extract_article(
        load("news_jsonld.html"), "https://www.harborledger.example/news/flood-barrier?utm_source=x"
    )


def test_news_page_yields_title_text_and_authors(news: ExtractedArticle) -> None:
    assert news.title == "City council approves riverside flood barrier after two-year review"
    assert words(news.text) >= 200
    assert "voted on Thursday evening" in news.text
    assert "Maria Okafor" in news.authors
    assert news.authors == ["Maria Okafor", "Daniel Reyes"]
    assert news.site_name == "Harbor Ledger"


def test_news_page_text_excludes_navigation_and_footer(news: ExtractedArticle) -> None:
    assert "All rights reserved" not in news.text
    assert "Sport" not in news.text


def test_json_ld_dates_win_over_meta_tags(news: ExtractedArticle) -> None:
    assert news.date_source == "jsonld"
    assert news.published_at == datetime(2024, 5, 17, 7, 15, tzinfo=UTC)
    assert news.modified_at == datetime(2024, 5, 18, 9, 0, tzinfo=UTC)


def test_canonical_url_comes_from_the_page(news: ExtractedArticle) -> None:
    assert news.canonical_url == (
        "https://www.harborledger.example/news/2024/05/17/flood-barrier-approved"
    )
    assert news.paywall_hint is False


# ---------------------------------------------------------------- plain blog page


def test_blog_without_metadata_still_yields_text_and_title() -> None:
    blog = extract_article(load("blog_plain.html"), "https://bakery.example/2023/11/04/wood-oven")
    assert blog.title == "Why our village bakery switched to a wood-fired oven"
    assert words(blog.text) >= 200
    assert "3 comments" not in blog.text
    assert blog.authors == ["Priya Nair"]
    assert blog.site_name is None
    assert blog.paywall_hint is False


def test_date_falls_back_to_the_url_path() -> None:
    blog = extract_article(load("blog_plain.html"), "https://bakery.example/2023/11/04/wood-oven")
    assert blog.date_source == "url"
    assert blog.published_at == datetime(2023, 11, 4, tzinfo=UTC)
    assert blog.canonical_url is None

    undated = extract_article(load("blog_plain.html"), "https://bakery.example/wood-oven")
    assert undated.date_source == "none"
    assert undated.published_at is None


# ---------------------------------------------------------------- paywall and script shell


def test_paywalled_page_is_flagged_and_keeps_its_teaser() -> None:
    html = load("paywalled.html")
    article = extract_article(html, "https://courier.example/airline-routes")
    assert article.paywall_hint is True
    assert "suspend six routes" in article.text
    assert article.date_source == "meta"
    assert article.published_at == datetime(2025, 1, 9, 6, 0, tzinfo=UTC)
    assert looks_paywalled(article, html) is True


def test_short_text_with_a_paywall_phrase_is_flagged_without_json_ld() -> None:
    html = "<html><body><article><p>The minister resigned on Monday.</p><p>Subscribe to continue reading.</p></article></body></html>"
    assert extract_article(html, "https://x.example/a").paywall_hint is True


def test_long_free_article_mentioning_subscriptions_is_not_flagged(news: ExtractedArticle) -> None:
    html = load("news_jsonld.html").replace(
        "</article>", "<p>Subscribe to continue reading our newsletter.</p></article>"
    )
    assert looks_paywalled(news, html) is False


def test_script_shell_yields_almost_no_text() -> None:
    shell = extract_article(load("js_shell.html"), "https://pulse.example/story/48213")
    assert words(shell.text) < 40
    assert shell.title == "Daily Pulse"
    assert shell.paywall_hint is False


@pytest.mark.parametrize("html", ["", "   ", "<html></html>", "not html at all", "<p>unclosed"])
def test_degenerate_input_does_not_raise(html: str) -> None:
    article = extract_article(html, "https://x.example/")
    assert words(article.text) < 40
    assert article.authors == []
    assert article.date_source == "none"


# ---------------------------------------------------------------- metadata details


def test_malformed_json_ld_is_ignored() -> None:
    html = (
        '<html><head><script type="application/ld+json">{"@type": "NewsArticle", oops</script>'
        '<meta property="article:published_time" content="2024-02-03T10:00:00Z">'
        "<title>Still parsed</title></head><body></body></html>"
    )
    meta = parse_metadata(html, "https://x.example/a")
    assert meta.title == "Still parsed"
    assert meta.date_source == "meta"


def test_unparseable_and_impossible_dates_are_dropped() -> None:
    html = (
        '<html><head><script type="application/ld+json">'
        '{"@type": "Article", "headline": "H", "datePublished": "next Tuesday", "author": "Solo Writer"}'
        "</script></head><body></body></html>"
    )
    meta = parse_metadata(html, "https://x.example/9999/99/99/a")
    assert meta.published_at is None
    assert meta.date_source == "none"
    assert meta.authors == ["Solo Writer"]


def test_title_suffix_with_the_site_name_is_trimmed() -> None:
    html = (
        "<html><head><title>Big story | Example News</title>"
        '<meta property="og:site_name" content="Example News"></head><body></body></html>'
    )
    assert parse_metadata(html, "https://x.example/a").title == "Big story"
