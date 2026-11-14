from pathlib import Path

import httpx
from fastapi import FastAPI

from fnd_api.api.errors import ApiError
from fnd_api.config import Settings
from fnd_api.main import create_app
from tests.integration.conftest import BASELINE_VERSION

PROBLEM_JSON = "application/problem+json"


async def test_healthz_is_ok(client: httpx.AsyncClient) -> None:
    response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_readyz_is_ready_when_a_model_is_loaded(client: httpx.AsyncClient) -> None:
    response = await client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "checks": {"models": True, "database": True}}


async def test_readyz_is_503_without_an_artifact(tmp_path: Path) -> None:
    empty = Settings(var_dir=tmp_path / "var", artifacts_dir=tmp_path / "none", log_json=False)
    app = create_app(empty)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http,
    ):
        ready = await http.get("/readyz")
        alive = await http.get("/healthz")
    assert ready.status_code == 503
    assert ready.json()["checks"]["models"] is False
    assert alive.status_code == 200


async def test_meta_lists_model_versions_providers_and_limits(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/v1/meta")
    assert response.status_code == 200
    body = response.json()
    assert body["models"] == {"baseline": BASELINE_VERSION}
    assert body["providers"]["external"] is False
    assert body["limits"] == {
        "rate_limit_analyze": "30/minute",
        "rate_limit_read": "120/minute",
        "max_text_chars": 100_000,
        "max_fetch_bytes": 5_000_000,
        "fetch_timeout_s": 8.0,
    }


async def test_unknown_route_is_problem_json_with_a_request_id(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/v1/nope")
    assert response.status_code == 404
    assert response.headers["content-type"] == PROBLEM_JSON
    body = response.json()
    assert body["code"] == "not_found"
    assert body["status"] == 404
    assert body["request_id"]
    assert response.headers["X-Request-ID"] == body["request_id"]


async def test_every_response_carries_a_request_id(client: httpx.AsyncClient) -> None:
    first = await client.get("/healthz")
    second = await client.get("/healthz")
    assert first.headers["X-Request-ID"] != second.headers["X-Request-ID"]


async def test_a_well_formed_incoming_request_id_is_kept(client: httpx.AsyncClient) -> None:
    kept = await client.get("/healthz", headers={"X-Request-ID": "abc-123_DEF"})
    replaced = await client.get("/healthz", headers={"X-Request-ID": "bad id\twith spaces" * 9})
    assert kept.headers["X-Request-ID"] == "abc-123_DEF"
    assert replaced.headers["X-Request-ID"] != "bad id\twith spaces" * 9


async def test_validation_error_is_invalid_input(app: FastAPI, client: httpx.AsyncClient) -> None:
    async def double(n: int) -> dict[str, int]:
        return {"n": n * 2}

    app.add_api_route("/_test/double/{n}", double)
    assert (await client.get("/_test/double/21")).json() == {"n": 42}

    response = await client.get("/_test/double/many")
    assert response.status_code == 422
    assert response.headers["content-type"] == PROBLEM_JSON
    body = response.json()
    assert body["code"] == "invalid_input"
    assert body["errors"][0]["loc"] == ["path", "n"]
    assert "input" not in body["errors"][0]
    assert body["request_id"] == response.headers["X-Request-ID"]


async def test_api_error_sets_status_and_code(app: FastAPI, client: httpx.AsyncClient) -> None:
    async def refuse() -> None:
        raise ApiError("content_too_short", 422, "The text has too few words to analyse.")

    app.add_api_route("/_test/refuse", refuse)
    response = await client.get("/_test/refuse")
    assert response.status_code == 422
    assert response.headers["content-type"] == PROBLEM_JSON
    body = response.json()
    assert body["code"] == "content_too_short"
    assert body["detail"] == "The text has too few words to analyse."
    assert body["title"] == "Unprocessable Entity"


async def test_unexpected_error_is_a_500_that_hides_the_cause(
    app: FastAPI, client: httpx.AsyncClient
) -> None:
    async def explode() -> None:
        raise RuntimeError("secret database password")

    app.add_api_route("/_test/explode", explode)
    response = await client.get("/_test/explode")
    assert response.status_code == 500
    assert response.headers["content-type"] == PROBLEM_JSON
    body = response.json()
    assert body["code"] == "internal"
    assert "secret" not in response.text
    assert body["request_id"] == response.headers["X-Request-ID"]


async def test_wrong_method_is_problem_json(client: httpx.AsyncClient) -> None:
    response = await client.post("/healthz")
    assert response.status_code == 405
    assert response.json()["code"] == "invalid_input"


def test_settings_derive_storage_paths_from_var_dir(tmp_path: Path) -> None:
    derived = Settings(var_dir=tmp_path)
    assert derived.cache_dir == tmp_path / "cache"
    assert derived.database_url == f"sqlite+aiosqlite:///{(tmp_path / 'fnd.db').as_posix()}"
    assert derived.redis_url is None
    assert derived.allow_private_urls is False


def test_settings_read_the_environment(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("FND_MAX_TEXT_CHARS", "500")
    monkeypatch.setenv("FND_DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    from_env = Settings(var_dir=tmp_path)
    assert from_env.max_text_chars == 500
    assert from_env.database_url == "sqlite+aiosqlite:///:memory:"
