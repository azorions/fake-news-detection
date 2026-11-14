from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import respx

from fnd_api.config import Settings
from fnd_api.ingest import FetchBlocked, FetchFailed, PayloadTooLarge
from fnd_api.ingest.fetcher import MAX_REDIRECTS, fetch_html

HTML = "<html><head><title>T</title></head><body><p>Hello world</p></body></html>"
HTML_HEADERS = {"content-type": "text/html; charset=utf-8"}


async def resolver(host: str, port: int) -> list[str]:
    return ["10.0.0.5"] if host.startswith("internal") else ["93.184.216.34"]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(var_dir=tmp_path, max_fetch_bytes=2_000, fetch_timeout_s=2.0)


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(follow_redirects=False) as http:
        yield http


@respx.mock
async def test_fetches_a_page(client: httpx.AsyncClient, settings: Settings) -> None:
    respx.get("https://news.example/story").respond(200, headers=HTML_HEADERS, text=HTML)
    page = await fetch_html("https://news.example/story", client, settings, resolver=resolver)
    assert page.html == HTML
    assert page.status == 200
    assert page.url == page.final_url == "https://news.example/story"
    assert page.fetched_at.tzinfo is not None


@respx.mock
async def test_follows_a_public_redirect(client: httpx.AsyncClient, settings: Settings) -> None:
    respx.get("http://news.example/old").respond(301, headers={"location": "/new?x=1"})
    respx.get("http://news.example/new?x=1").respond(
        302, headers={"location": "https://www.news.example/final"}
    )
    respx.get("https://www.news.example/final").respond(200, headers=HTML_HEADERS, text=HTML)
    page = await fetch_html("http://news.example/old", client, settings, resolver=resolver)
    assert page.url == "http://news.example/old"
    assert page.final_url == "https://www.news.example/final"
    assert page.html == HTML


@respx.mock
@pytest.mark.parametrize(
    "target",
    ["http://127.0.0.1/", "http://internal.example/secret", "http://169.254.169.254/latest", "ftp://x.example/"],
)  # fmt: skip
async def test_redirect_to_a_private_address_is_blocked_at_the_hop(
    client: httpx.AsyncClient, settings: Settings, target: str
) -> None:
    respx.get("http://news.example/go").respond(302, headers={"location": target})
    private = respx.route(
        host__regex=r"127\.0\.0\.1|internal\.example|169\.254\.169\.254|x\.example"
    ).respond(200, headers=HTML_HEADERS, text="secret")
    with pytest.raises(FetchBlocked):
        await fetch_html("http://news.example/go", client, settings, resolver=resolver)
    assert private.call_count == 0


@respx.mock
async def test_private_start_url_is_never_requested(
    client: httpx.AsyncClient, settings: Settings
) -> None:
    private = respx.get("http://internal.example/").respond(200, headers=HTML_HEADERS, text="x")
    with pytest.raises(FetchBlocked):
        await fetch_html("http://internal.example/", client, settings, resolver=resolver)
    assert private.call_count == 0


@respx.mock
async def test_redirect_loops_end(client: httpx.AsyncClient, settings: Settings) -> None:
    loop = respx.get("http://news.example/loop").respond(302, headers={"location": "/loop"})
    with pytest.raises(FetchFailed, match="redirect"):
        await fetch_html("http://news.example/loop", client, settings, resolver=resolver)
    assert loop.call_count == MAX_REDIRECTS + 1


@respx.mock
async def test_declared_oversize_body_is_refused(
    client: httpx.AsyncClient, settings: Settings
) -> None:
    respx.get("http://news.example/big").respond(200, headers=HTML_HEADERS, text="x" * 5_000)
    with pytest.raises(PayloadTooLarge):
        await fetch_html("http://news.example/big", client, settings, resolver=resolver)


@respx.mock
async def test_streamed_oversize_body_is_cut_off(
    client: httpx.AsyncClient, settings: Settings
) -> None:
    sent = 0

    async def endless() -> AsyncIterator[bytes]:
        nonlocal sent
        for _ in range(1_000):
            sent += 1
            yield b"x" * 500

    respx.get("http://news.example/stream").mock(
        side_effect=lambda request: httpx.Response(200, headers=HTML_HEADERS, content=endless())
    )
    with pytest.raises(PayloadTooLarge):
        await fetch_html("http://news.example/stream", client, settings, resolver=resolver)
    assert sent < 20  # stopped reading soon after the cap, not after the whole body


@respx.mock
@pytest.mark.parametrize(
    ("status", "headers"),
    [
        (200, {"content-type": "application/pdf"}),
        (200, {"content-type": "image/png"}),
        (500, HTML_HEADERS),
        (404, HTML_HEADERS),
        (403, HTML_HEADERS),
        (302, {}),  # redirect without a Location
    ],
)
async def test_unusable_responses_fail(
    client: httpx.AsyncClient, settings: Settings, status: int, headers: dict[str, str]
) -> None:
    respx.get("http://news.example/x").respond(status, headers=headers, text="body")
    with pytest.raises(FetchFailed) as failed:
        await fetch_html("http://news.example/x", client, settings, resolver=resolver)
    assert failed.value.code == "fetch_failed"


@respx.mock
@pytest.mark.parametrize("content_type", ["application/xhtml+xml", "text/plain", "TEXT/HTML"])
async def test_text_content_types_are_accepted(
    client: httpx.AsyncClient, settings: Settings, content_type: str
) -> None:
    respx.get("http://news.example/x").respond(
        200, headers={"content-type": content_type}, text=HTML
    )
    assert (await fetch_html("http://news.example/x", client, settings, resolver=resolver)).html


@respx.mock
@pytest.mark.parametrize(
    "error", [httpx.ConnectTimeout("slow"), httpx.ConnectError("refused"), httpx.ReadError("reset")]
)
async def test_network_errors_fail_without_leaking_details(
    client: httpx.AsyncClient, settings: Settings, error: Exception
) -> None:
    respx.get("http://news.example/x").mock(side_effect=error)
    with pytest.raises(FetchFailed) as failed:
        await fetch_html("http://news.example/x", client, settings, resolver=resolver)
    assert "refused" not in str(failed.value)


@respx.mock
async def test_charset_from_header_or_meta_tag_is_used(
    client: httpx.AsyncClient, settings: Settings
) -> None:
    respx.get("http://news.example/latin").respond(
        200,
        headers={"content-type": "text/html; charset=iso-8859-1"},
        content="café".encode("latin-1"),
    )
    meta_page = '<html><head><meta charset="windows-1252"></head><body>“quoted”</body></html>'
    respx.get("http://news.example/meta").respond(
        200, headers={"content-type": "text/html"}, content=meta_page.encode("cp1252")
    )
    respx.get("http://news.example/bad").respond(
        200, headers={"content-type": "text/html; charset=utf-8"}, content=b"ok \xff\xfe end"
    )
    assert (
        await fetch_html("http://news.example/latin", client, settings, resolver=resolver)
    ).html == "café"
    assert (
        "“quoted”"
        in (await fetch_html("http://news.example/meta", client, settings, resolver=resolver)).html
    )
    assert (
        await fetch_html("http://news.example/bad", client, settings, resolver=resolver)
    ).html.startswith("ok ")
