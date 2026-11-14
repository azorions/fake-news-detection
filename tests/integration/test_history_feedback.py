from typing import Any

import httpx
import respx

from fnd_api.config import Settings
from tests.fixtures.synthetic import NEUTRAL_TEXT, SENSATIONAL_TEXT
from tests.integration.conftest import ClientFactory

ANALYZE = "/api/v1/analyze"
SCANS = "/api/v1/scans"
PROBLEM_JSON = "application/problem+json"


async def analyse(client: httpx.AsyncClient, **payload: Any) -> dict[str, Any]:
    response = await client.post(ANALYZE, json=payload)
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


# ---------------------------------------------------------------- history


async def test_empty_history(client: httpx.AsyncClient) -> None:
    response = await client.get(SCANS)
    assert response.status_code == 200
    assert response.json() == {"items": [], "next_cursor": None}


async def test_analysed_scan_is_listed_first_and_round_trips(client: httpx.AsyncClient) -> None:
    older = await analyse(client, text=NEUTRAL_TEXT)
    newer = await analyse(client, text=SENSATIONAL_TEXT, title="A shocking title")

    listing = (await client.get(SCANS)).json()
    assert [item["scan_id"] for item in listing["items"]] == [newer["scan_id"], older["scan_id"]]
    first = listing["items"][0]
    assert first["verdict"] == "high_risk"
    assert first["title"] == "A shocking title"
    assert first["input_type"] == "headline"
    assert first["feedback_count"] == 0
    assert first["risk_score"] == newer["risk_score"]
    assert listing["next_cursor"] is None

    fetched = await client.get(f"{SCANS}/{newer['scan_id']}")
    assert fetched.status_code == 200
    assert fetched.json() == newer


async def test_pagination_walks_every_scan_once(client: httpx.AsyncClient) -> None:
    first = await analyse(client, text=NEUTRAL_TEXT)
    second = await analyse(client, text=SENSATIONAL_TEXT)

    page_one = (await client.get(SCANS, params={"limit": 1})).json()
    assert [item["scan_id"] for item in page_one["items"]] == [second["scan_id"]]
    assert page_one["next_cursor"]
    page_two = (
        await client.get(SCANS, params={"limit": 1, "cursor": page_one["next_cursor"]})
    ).json()
    assert [item["scan_id"] for item in page_two["items"]] == [first["scan_id"]]
    assert page_two["next_cursor"] is None


@respx.mock
async def test_filters_by_verdict_domain_and_query(client: httpx.AsyncClient) -> None:
    html = (
        "<html><head><title>Budget approved</title></head><body><article>"
        + f"<p>{NEUTRAL_TEXT}</p>" * 3
        + "</article></body></html>"
    )
    respx.get("https://www.ledger.example/budget").respond(
        200, headers={"content-type": "text/html"}, text=html
    )
    from_url = await analyse(client, url="https://www.ledger.example/budget")
    risky = await analyse(client, text=SENSATIONAL_TEXT, title="Miracle cure exposed")

    async def ids(**params: Any) -> list[str]:
        return [
            item["scan_id"] for item in (await client.get(SCANS, params=params)).json()["items"]
        ]

    assert await ids(verdict="high_risk") == [risky["scan_id"]]
    assert await ids(verdict="reliable") == [from_url["scan_id"]]
    assert await ids(domain="ledger.example") == [from_url["scan_id"]]
    assert await ids(domain="LEDGER.example") == [from_url["scan_id"]]
    assert await ids(q="miracle") == [risky["scan_id"]]
    assert await ids(q="budget") == [from_url["scan_id"]]
    assert await ids(q="no such thing") == []


async def test_invalid_list_parameters_are_rejected(client: httpx.AsyncClient) -> None:
    for params in ({"limit": 0}, {"limit": 101}, {"verdict": "fake"}, {"cursor": "garbage"}):
        response = await client.get(SCANS, params=params)
        assert response.status_code == 422, params
        assert response.headers["content-type"] == PROBLEM_JSON
        assert response.json()["code"] == "invalid_input"


async def test_unknown_scan_is_not_found(client: httpx.AsyncClient) -> None:
    for response in (
        await client.get(f"{SCANS}/01HZZZZZZZZZZZZZZZZZZZZZZZ"),
        await client.delete(f"{SCANS}/01HZZZZZZZZZZZZZZZZZZZZZZZ"),
        await client.post(f"{SCANS}/01HZZZZZZZZZZZZZZZZZZZZZZZ/feedback", json={"rating": "agree"}),
    ):
        assert response.status_code == 404
        assert response.headers["content-type"] == PROBLEM_JSON
        assert response.json()["code"] == "not_found"


# ---------------------------------------------------------------- feedback


async def test_feedback_is_created_and_counted(client: httpx.AsyncClient) -> None:
    scan = await analyse(client, text=SENSATIONAL_TEXT)
    url = f"{SCANS}/{scan['scan_id']}/feedback"

    created = await client.post(
        url,
        json={"rating": "disagree", "corrected_verdict": "reliable", "comment": "It is satire."},
    )
    assert created.status_code == 201
    body = created.json()
    assert body["scan_id"] == scan["scan_id"]
    assert body["rating"] == "disagree"
    assert body["corrected_verdict"] == "reliable"
    assert body["target_type"] == "verdict"
    assert body["comment"] == "It is satire."
    assert body["id"] and body["created_at"]

    span_id = scan["highlights"][0]["id"]
    on_span = await client.post(
        url, json={"rating": "agree", "target_type": "highlight", "target_id": span_id}
    )
    assert on_span.status_code == 201

    listing = (await client.get(SCANS)).json()
    assert listing["items"][0]["feedback_count"] == 2


async def test_invalid_feedback_is_rejected(client: httpx.AsyncClient) -> None:
    scan = await analyse(client, text=SENSATIONAL_TEXT)
    url = f"{SCANS}/{scan['scan_id']}/feedback"
    for payload in (
        {"rating": "disagree", "corrected_verdict": "fake"},
        {"rating": "maybe"},
        {},
        {"rating": "agree", "comment": "x" * 2001},
        {"rating": "agree", "target_type": "paragraph"},
    ):
        response = await client.post(url, json=payload)
        assert response.status_code == 422, payload
        assert response.json()["code"] == "invalid_input"
    assert (await client.get(SCANS)).json()["items"][0]["feedback_count"] == 0


# ---------------------------------------------------------------- delete


async def test_delete_removes_the_scan_its_feedback_and_its_cached_result(
    client: httpx.AsyncClient,
) -> None:
    scan = await analyse(client, text=SENSATIONAL_TEXT)
    other = await analyse(client, text=NEUTRAL_TEXT)
    await client.post(f"{SCANS}/{scan['scan_id']}/feedback", json={"rating": "agree"})

    deleted = await client.delete(f"{SCANS}/{scan['scan_id']}")
    assert deleted.status_code == 204
    assert deleted.content == b""
    assert (await client.get(f"{SCANS}/{scan['scan_id']}")).status_code == 404
    assert (await client.delete(f"{SCANS}/{scan['scan_id']}")).status_code == 404
    assert [item["scan_id"] for item in (await client.get(SCANS)).json()["items"]] == [
        other["scan_id"]
    ]

    # The deleted text is gone from the cache too: the same input is analysed afresh.
    again = await analyse(client, text=SENSATIONAL_TEXT)
    assert again["cached"] is False
    assert again["scan_id"] != scan["scan_id"]


async def test_reads_are_rate_limited_separately(
    settings: Settings, make_client: ClientFactory
) -> None:
    limited = settings.model_copy(update={"rate_limit_read": "2/minute"})
    async with make_client(limited) as (_, http):
        statuses = [(await http.get(SCANS)).status_code for _ in range(3)]
        analysis = await http.post(ANALYZE, json={"text": NEUTRAL_TEXT})
    assert statuses == [200, 200, 429]
    assert analysis.status_code == 200
