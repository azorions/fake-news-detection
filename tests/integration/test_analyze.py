import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi import FastAPI

from fnd_api.config import Settings
from fnd_api.db.repositories import ScanRepository
from tests.fixtures.synthetic import NEUTRAL_TEXT, SENSATIONAL_TEXT
from tests.integration.conftest import BASELINE_VERSION, ClientFactory

FIXTURES = Path(__file__).parents[2] / "data" / "fixtures" / "html"
HTML = {"content-type": "text/html; charset=utf-8"}
ANALYZE = "/api/v1/analyze"
SPANISH = (
    "El ayuntamiento votó el jueves por la noche a favor de construir una barrera contra las "
    "inundaciones en la orilla oriental del río, después de una revisión de más de dos años."
)


def page(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def assert_offsets_are_exact(body: dict[str, Any]) -> None:
    text = body["article"]["text"]
    assert body["highlights"], "expected at least one highlight"
    for span in body["highlights"]:
        assert 0 <= span["start"] < span["end"] <= len(text)
        assert text[span["start"] : span["end"]].strip()
    tokens = [token for attribution in body["attributions"] for token in attribution["tokens"]]
    assert tokens, "expected token attributions"
    for token in tokens:
        assert text[token["start"] : token["end"]].lower() == token["token"]


def scans_of(app: FastAPI) -> ScanRepository:
    return ScanRepository(app.state.sessions)


# ---------------------------------------------------------------- pasted text


async def test_sensational_text_is_high_risk_with_exact_offsets(
    app: FastAPI, client: httpx.AsyncClient
) -> None:
    response = await client.post(ANALYZE, json={"text": SENSATIONAL_TEXT})
    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "high_risk"
    assert body["risk_score"] > 0.65
    assert body["article"]["text"] == SENSATIONAL_TEXT
    assert body["article"]["input_type"] == "headline"  # under 40 words
    assert body["article"]["extraction"] == "user_text"
    assert body["article"]["language"] == "en"
    assert body["classifier"]["model"] == "baseline"
    assert body["classifier"]["version"] == BASELINE_VERSION
    assert body["classifier"]["calibration"] == "isotonic"
    assert body["classifier"]["satire_score"] is None
    assert body["models"] == {"baseline": BASELINE_VERSION}
    assert body["cached"] is False
    assert_offsets_are_exact(body)
    kinds = {span["kind"] for span in body["highlights"]}
    assert "sensational" in kinds
    assert all(span["origin"] == "rule" for span in body["highlights"])
    assert body["attributions"][0]["method"] == "lexical_linear"
    assert [stage["name"] for stage in body["stages"]] == [
        "ingest", "source", "classifier", "signals", "explanations", "final_verdict",
    ]  # fmt: skip
    assert body["score_breakdown"][0]["name"] == "classifier"
    assert body["summary"]

    stored = await scans_of(app).get(body["scan_id"])
    assert stored is not None
    assert stored.verdict == "high_risk"


async def test_long_neutral_text_is_reliable_with_usable_confidence(
    client: httpx.AsyncClient,
) -> None:
    body = (await client.post(ANALYZE, json={"text": NEUTRAL_TEXT * 2})).json()
    assert body["verdict"] == "reliable"
    assert body["article"]["input_type"] == "text"
    assert body["confidence"] > 0.6
    assert body["confidence_level"] in ("medium", "high")
    assert body["tags"] == []


async def test_short_text_gets_low_confidence(client: httpx.AsyncClient) -> None:
    body = (await client.post(ANALYZE, json={"text": SENSATIONAL_TEXT})).json()
    assert body["confidence"] <= 0.6
    assert body["confidence_level"] == "low"


async def test_offsets_hold_for_emoji_combining_marks_and_crlf(client: httpx.AsyncClient) -> None:
    text = (
        "😱😱 SHOCKING:\r\nthe government is hiding the truth about vaccines!!!\r\n\r\n"
        "Café owners 👩‍👩‍👧 say “wake up, people” — experts say it is a miracle cure."
    )
    response = await client.post(ANALYZE, json={"text": text, "title": "BOMBSHELL 🔥 report"})
    assert response.status_code == 200
    body = response.json()
    assert "\r" not in body["article"]["text"]
    assert_offsets_are_exact(body)
    highlighted = {body["article"]["text"][s["start"] : s["end"]] for s in body["highlights"]}
    assert {"SHOCKING", "!!!", "experts say", "miracle cure"} <= highlighted
    assert body["metadata"]["title"] == "BOMBSHELL 🔥 report"


async def test_identical_request_is_served_from_the_cache(client: httpx.AsyncClient) -> None:
    first = (await client.post(ANALYZE, json={"text": SENSATIONAL_TEXT})).json()
    second = (await client.post(ANALYZE, json={"text": SENSATIONAL_TEXT})).json()
    assert first["cached"] is False
    assert second["cached"] is True
    for field in ("scan_id", "verdict", "risk_score", "highlights", "created_at"):
        assert second[field] == first[field]

    fresh = (
        await client.post(ANALYZE, json={"text": SENSATIONAL_TEXT, "options": {"use_cache": False}})
    ).json()
    assert fresh["cached"] is False
    assert fresh["scan_id"] != first["scan_id"]

    other_depth = (
        await client.post(ANALYZE, json={"text": SENSATIONAL_TEXT, "options": {"depth": "quick"}})
    ).json()
    assert other_depth["cached"] is False


async def test_store_false_keeps_the_scan_out_of_history(
    app: FastAPI, client: httpx.AsyncClient
) -> None:
    body = (
        await client.post(ANALYZE, json={"text": SENSATIONAL_TEXT, "options": {"store": False}})
    ).json()
    assert await scans_of(app).get(body["scan_id"]) is None

    again = (await client.post(ANALYZE, json={"text": SENSATIONAL_TEXT})).json()
    assert again["cached"] is True
    assert await scans_of(app).get(again["scan_id"]) is not None  # stored on the later request


async def test_two_identical_requests_at_once_both_succeed(
    app: FastAPI, client: httpx.AsyncClient
) -> None:
    responses = await asyncio.gather(
        *(client.post(ANALYZE, json={"text": NEUTRAL_TEXT}) for _ in range(4))
    )
    assert [r.status_code for r in responses] == [200, 200, 200, 200]
    listed, _ = await scans_of(app).list()
    assert {item.scan_id for item in listed} == {r.json()["scan_id"] for r in responses}


# ---------------------------------------------------------------- refused input


@pytest.mark.parametrize(
    ("payload", "status", "code"),
    [
        ({"text": "Too short"}, 422, "content_too_short"),
        ({"text": "!!! ??? ..."}, 422, "content_too_short"),
        ({"text": SPANISH}, 422, "unsupported_language"),
        ({}, 422, "invalid_input"),
        ({"text": "some text here", "url": "https://example.com/a"}, 422, "invalid_input"),
        ({"text": "some text here", "extra": 1}, 422, "invalid_input"),
        ({"url": "http://127.0.0.1/"}, 422, "fetch_blocked"),
        ({"url": "http://169.254.169.254/latest/meta-data"}, 422, "fetch_blocked"),
        ({"url": "http://[::1]/"}, 422, "fetch_blocked"),
        ({"url": "http://user:pw@example.com/"}, 422, "fetch_blocked"),
    ],
)
async def test_refused_input(
    client: httpx.AsyncClient, payload: dict[str, Any], status: int, code: str
) -> None:
    response = await client.post(ANALYZE, json=payload)
    assert response.status_code == status
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json()["code"] == code
    assert response.json()["request_id"]


async def test_refused_input_is_not_stored(app: FastAPI, client: httpx.AsyncClient) -> None:
    await client.post(ANALYZE, json={"text": SPANISH})
    await client.post(ANALYZE, json={"url": "http://127.0.0.1/"})
    assert (await scans_of(app).list())[0] == []


async def test_text_over_the_configured_limit_is_too_large(
    settings: Settings, make_client: ClientFactory
) -> None:
    small = settings.model_copy(update={"max_text_chars": 50})
    async with make_client(small) as (_, http):
        response = await http.post(ANALYZE, json={"text": NEUTRAL_TEXT})
    assert response.status_code == 413
    assert response.json()["code"] == "payload_too_large"


async def test_huge_request_body_is_refused_before_it_is_parsed(client: httpx.AsyncClient) -> None:
    response = await client.post(
        ANALYZE, content=b'{"text": "' + b"x" * 2_100_000 + b'"}', headers={"content-type": "application/json"}
    )  # fmt: skip
    assert response.status_code == 413
    assert response.json()["code"] == "payload_too_large"


# ---------------------------------------------------------------- URLs


@respx.mock
async def test_url_is_fetched_extracted_and_analysed(
    app: FastAPI, client: httpx.AsyncClient
) -> None:
    route = respx.get("https://www.harborledger.example/news/flood-barrier").respond(
        200, headers=HTML, text=page("news_jsonld.html")
    )
    payload = {"url": "https://www.harborledger.example/news/flood-barrier"}
    response = await client.post(ANALYZE, json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["article"]["input_type"] == "url"
    assert body["article"]["extraction"] == "full"
    assert body["article"]["url"] == "https://www.harborledger.example/news/flood-barrier"
    assert body["article"]["canonical_url"].endswith("/news/2024/05/17/flood-barrier-approved")
    assert body["article"]["word_count"] >= 200
    assert body["metadata"]["title"].startswith("City council approves riverside flood barrier")
    assert body["metadata"]["authors"] == ["Maria Okafor", "Daniel Reyes"]
    assert body["metadata"]["site_name"] == "Harbor Ledger"
    assert body["metadata"]["published_at"].startswith("2024-05-17T07:15:00")
    assert body["metadata"]["date_source"] == "jsonld"
    assert body["source"]["domain"] == "harborledger.example"
    assert body["source"]["known"] is False
    assert body["verdict"] == "reliable"
    assert body["tags"] == []

    again = (await client.post(ANALYZE, json=payload)).json()
    assert again["cached"] is True
    assert route.call_count == 1  # a cached result does not fetch the page again

    listed, _ = await scans_of(app).list(domain="harborledger.example")
    assert [item.title for item in listed] == [body["metadata"]["title"]]


@respx.mock
async def test_script_shell_page_is_unverified_not_guessed(client: httpx.AsyncClient) -> None:
    respx.get("https://pulse.example/story/48213").respond(
        200, headers=HTML, text=page("js_shell.html")
    )
    body = (await client.post(ANALYZE, json={"url": "https://pulse.example/story/48213"})).json()
    assert body["verdict"] == "unverified"
    assert "partial_text" in body["tags"]
    assert body["article"]["extraction"] == "partial"
    assert body["confidence"] <= 0.6
    assert body["confidence_level"] == "low"
    assert "paste" in body["summary"].lower()
    assert any(component["name"] == "guard" for component in body["score_breakdown"])


@respx.mock
async def test_paywalled_page_is_tagged_and_capped(client: httpx.AsyncClient) -> None:
    respx.get("https://courier.example/airline").respond(
        200, headers=HTML, text=page("paywalled.html")
    )
    body = (await client.post(ANALYZE, json={"url": "https://courier.example/airline"})).json()
    assert "partial_text" in body["tags"]
    assert body["article"]["extraction"] == "partial"
    assert body["confidence"] <= 0.6
    assert "suspend six routes" in body["article"]["text"]


@respx.mock
async def test_redirect_to_a_private_address_is_blocked(client: httpx.AsyncClient) -> None:
    respx.get("https://short.example/x").respond(
        302, headers={"location": "http://127.0.0.1:8000/admin"}
    )
    private = respx.get("http://127.0.0.1:8000/admin").respond(200, headers=HTML, text="secret")
    response = await client.post(ANALYZE, json={"url": "https://short.example/x"})
    assert response.status_code == 422
    assert response.json()["code"] == "fetch_blocked"
    assert private.call_count == 0


@respx.mock
@pytest.mark.parametrize(
    ("upstream", "status", "code"),
    [
        (httpx.Response(500, headers=HTML, text="oops"), 502, "fetch_failed"),
        (httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF"), 502, "fetch_failed"),
        (httpx.Response(200, headers=HTML, text="x" * 6_000_000), 413, "payload_too_large"),
    ],
)  # fmt: skip
async def test_unusable_pages_fail_cleanly(
    client: httpx.AsyncClient, upstream: httpx.Response, status: int, code: str
) -> None:
    respx.get("https://news.example/a").mock(return_value=upstream)
    response = await client.post(ANALYZE, json={"url": "https://news.example/a"})
    assert response.status_code == status
    assert response.json()["code"] == code


@respx.mock
async def test_non_english_page_is_unsupported(client: httpx.AsyncClient) -> None:
    html = f"<html><head><title>Noticias</title></head><body><article>{'<p>' + SPANISH + '</p>' * 1}<p>{SPANISH}</p><p>{SPANISH}</p></article></body></html>"  # fmt: skip
    respx.get("https://noticias.example/a").respond(200, headers=HTML, text=html)
    response = await client.post(ANALYZE, json={"url": "https://noticias.example/a"})
    assert response.status_code == 422
    assert response.json()["code"] == "unsupported_language"


# ---------------------------------------------------------------- limits and configuration


async def test_fourth_request_in_a_minute_is_rate_limited(
    settings: Settings, make_client: ClientFactory
) -> None:
    limited = settings.model_copy(update={"rate_limit_analyze": "3/minute"})
    async with make_client(limited) as (_, http):
        statuses = [
            (await http.post(ANALYZE, json={"text": NEUTRAL_TEXT})).status_code for _ in range(3)
        ]
        blocked = await http.post(ANALYZE, json={"text": NEUTRAL_TEXT})
        health = await http.get("/healthz")
    assert statuses == [200, 200, 200]
    assert blocked.status_code == 429
    assert blocked.headers["content-type"] == "application/problem+json"
    assert blocked.json()["code"] == "rate_limited"
    assert int(blocked.headers["Retry-After"]) >= 1
    assert health.status_code == 200  # other endpoints are not affected


async def test_without_a_model_analysis_is_unavailable(
    tmp_path: Path, make_client: ClientFactory
) -> None:
    no_model = Settings(var_dir=tmp_path / "v", artifacts_dir=tmp_path / "none", log_json=False)
    async with make_client(no_model) as (_, http):
        response = await http.post(ANALYZE, json={"text": NEUTRAL_TEXT})
    assert response.status_code == 503
    assert response.json()["code"] == "internal"


async def test_private_mode_stores_no_text_and_does_not_cache(
    settings: Settings, make_client: ClientFactory
) -> None:
    private = settings.model_copy(update={"store_input_text": False})
    async with make_client(private) as (application, http):
        first = (await http.post(ANALYZE, json={"text": SENSATIONAL_TEXT})).json()
        second = (await http.post(ANALYZE, json={"text": SENSATIONAL_TEXT})).json()
        stored = await ScanRepository(application.state.sessions, store_input_text=False).get(
            first["scan_id"]
        )
    assert first["article"]["text"] == SENSATIONAL_TEXT  # the caller still gets the full result
    assert second["cached"] is False
    assert stored is not None
    assert stored.article.text == ""
    assert stored.highlights == []
