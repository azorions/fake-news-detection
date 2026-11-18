"""The API with a transformer artifact: it gives the verdict, the baseline stands in when it
cannot, and the satire scorer only adds a tag.

The transformer is the committed toy artifact in ``tests/fixtures/tiny_transformer``.
"""

import shutil
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from fnd_api.config import Settings
from fnd_ml.common.artifacts import register, resolve
from fnd_ml.models.baseline import BaselineConfig, BaselineModel
from tests.fixtures.synthetic import NEUTRAL_TEXT, SENSATIONAL_TEXT, make_corpus
from tests.integration.conftest import BASELINE_VERSION, ClientFactory

ANALYZE = "/api/v1/analyze"
TINY = Path("tests/fixtures/tiny_transformer")
TRANSFORMER_VERSION = "v1-tinyfix"
SATIRE_VERSION = "v-satire-test"
SATIRE_HEADLINE = "Area man heroically declares himself mayor of the couch"
REAL_ARTIFACTS = Path("ml/artifacts")


def _add_transformer(root: Path) -> Path:
    directory = root / "transformer" / TRANSFORMER_VERSION
    shutil.copytree(TINY, directory)
    register(root, "transformer", TRANSFORMER_VERSION, f"transformer/{TRANSFORMER_VERSION}")
    return directory


def _add_satire(root: Path) -> None:
    """A scorer that knows one kind of satire: headlines about an "area man" and his couch."""
    plain = make_corpus(160, seed=5)["text"].tolist()
    satire = [f"{SATIRE_HEADLINE} for the {index}th time this week" for index in range(160)]
    texts, labels = plain + satire, [0] * len(plain) + [1] * len(satire)
    model = BaselineModel(BaselineConfig(n_features=2**14)).fit(texts, labels)
    model.calibrate(texts, labels)
    directory = root / "satire" / SATIRE_VERSION
    model.save(directory, {"version": SATIRE_VERSION}, kind="satire")
    register(root, "satire", SATIRE_VERSION, f"satire/{SATIRE_VERSION}")


def _copy(artifacts_dir: Path, target: Path) -> Path:
    return Path(shutil.copytree(artifacts_dir, target))


@pytest.fixture(scope="module")
def full_artifacts(artifacts_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Baseline, transformer and satire scorer."""
    root = _copy(artifacts_dir, tmp_path_factory.mktemp("full") / "artifacts")
    _add_transformer(root)
    _add_satire(root)
    return root


@pytest.fixture
def settings(tmp_path: Path, full_artifacts: Path) -> Settings:
    return Settings(var_dir=tmp_path / "var", artifacts_dir=full_artifacts, log_json=False)


def _stage(body: dict[str, Any], name: str) -> dict[str, Any]:
    (stage,) = [stage for stage in body["stages"] if stage["name"] == name]
    return stage


async def _analyze(client: httpx.AsyncClient, **payload: Any) -> dict[str, Any]:
    response = await client.post(ANALYZE, json=payload)
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


# ---------------------------------------------------------------- the transformer answers


async def test_meta_reports_the_transformer(client: httpx.AsyncClient) -> None:
    models = (await client.get("/api/v1/meta")).json()["models"]

    assert models == {
        "baseline": BASELINE_VERSION,
        "transformer": TRANSFORMER_VERSION,
        "satire": SATIRE_VERSION,
    }


async def test_transformer_gives_the_verdict(client: httpx.AsyncClient) -> None:
    body = await _analyze(client, text=SENSATIONAL_TEXT)

    assert body["classifier"]["model"] == "tiny-fixture"
    assert body["classifier"]["version"] == TRANSFORMER_VERSION
    assert body["classifier"]["calibration"] == "temperature"
    assert body["classifier"]["p_unreliable"] < body["classifier"]["p_unreliable_raw"]
    assert body["verdict"] == "high_risk"
    assert body["risk_score"] == body["classifier"]["p_unreliable"]
    assert _stage(body, "classifier")["status"] == "ok"
    assert _stage(body, "classifier")["detail"] is None
    assert "tiny-fixture" in body["score_breakdown"][0]["detail"]
    assert body["models"]["transformer"] == TRANSFORMER_VERSION
    # The word weights still come from the baseline, and say so.
    assert body["attributions"][0]["model"] == "baseline"
    assert body["attributions"][0]["tokens"]
    assert _stage(body, "explanations")["status"] == "ok"


async def test_reporting_is_reliable_through_the_transformer(client: httpx.AsyncClient) -> None:
    body = await _analyze(client, text=NEUTRAL_TEXT * 2)

    assert body["classifier"]["model"] == "tiny-fixture"
    assert body["verdict"] == "reliable"


async def test_a_very_long_text_is_truncated_not_rejected(client: httpx.AsyncClient) -> None:
    text = ("The finance ministry said on Tuesday that the budget was approved. " * 1500)[:99_000]

    body = await _analyze(client, text=text)

    assert body["classifier"]["model"] == "tiny-fixture"
    assert body["classifier"]["truncated"] is True
    assert body["verdict"] == "reliable"


# ---------------------------------------------------------------- the baseline stands in


async def test_without_the_artifact_the_baseline_answers(
    make_client: ClientFactory, artifacts_dir: Path, tmp_path: Path
) -> None:
    baseline_only = Settings(var_dir=tmp_path / "var", artifacts_dir=artifacts_dir, log_json=False)

    async with make_client(baseline_only) as (_, client):
        body = await _analyze(client, text=SENSATIONAL_TEXT)
        ready = await client.get("/readyz")

    assert ready.status_code == 200
    assert body["classifier"]["model"] == "baseline"
    assert body["classifier"]["calibration"] == "isotonic"
    assert _stage(body, "classifier")["status"] == "degraded"
    assert _stage(body, "classifier")["detail"] == "transformer model not available; baseline used"
    assert body["models"] == {"baseline": BASELINE_VERSION}


async def test_damaged_artifact_does_not_stop_the_service(
    make_client: ClientFactory, full_artifacts: Path, tmp_path: Path
) -> None:
    root = _copy(full_artifacts, tmp_path / "artifacts")
    model = root / "transformer" / TRANSFORMER_VERSION / "model.onnx"
    model.write_bytes(model.read_bytes()[:100])
    damaged = Settings(var_dir=tmp_path / "var", artifacts_dir=root, log_json=False)

    async with make_client(damaged) as (_, client):
        ready = await client.get("/readyz")
        models = (await client.get("/api/v1/meta")).json()["models"]
        body = await _analyze(client, text=SENSATIONAL_TEXT)

    assert ready.status_code == 200
    assert "transformer" not in models
    assert body["classifier"]["model"] == "baseline"
    assert _stage(body, "classifier")["status"] == "degraded"
    assert "baseline used" in _stage(body, "classifier")["detail"]
    assert body["verdict"] == "high_risk"


async def test_runtime_failure_falls_back(
    app: FastAPI, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(title: str | None, text: str) -> None:
        raise RuntimeError("the runtime gave up")

    monkeypatch.setattr(app.state.models.transformer, "predict", broken)

    body = await _analyze(client, text=SENSATIONAL_TEXT)

    assert body["classifier"]["model"] == "baseline"
    assert body["classifier"]["version"] == BASELINE_VERSION
    assert _stage(body, "classifier")["status"] == "degraded"
    assert _stage(body, "classifier")["detail"] == "transformer model failed; baseline used"
    assert body["verdict"] == "high_risk"
    assert "the runtime gave up" not in str(body)


async def test_without_a_baseline_the_transformer_answers_and_word_weights_are_skipped(
    make_client: ClientFactory, tmp_path: Path
) -> None:
    root = tmp_path / "artifacts"
    _add_transformer(root)
    transformer_only = Settings(var_dir=tmp_path / "var", artifacts_dir=root, log_json=False)

    async with make_client(transformer_only) as (_, client):
        ready = await client.get("/readyz")
        body = await _analyze(client, text=SENSATIONAL_TEXT)

    assert ready.status_code == 200
    assert body["classifier"]["model"] == "tiny-fixture"
    assert body["verdict"] == "high_risk"
    assert body["attributions"] == []
    assert _stage(body, "explanations")["status"] == "skipped"
    assert "baseline" in _stage(body, "explanations")["detail"]


async def test_result_is_not_served_from_the_cache_after_a_model_change(
    make_client: ClientFactory, artifacts_dir: Path, full_artifacts: Path, tmp_path: Path
) -> None:
    var_dir = tmp_path / "var"  # both applications share the cache and the database
    before = Settings(var_dir=var_dir, artifacts_dir=artifacts_dir, log_json=False)
    after = Settings(var_dir=var_dir, artifacts_dir=full_artifacts, log_json=False)

    async with make_client(before) as (_, client):
        first = await _analyze(client, text=SENSATIONAL_TEXT)
        assert (await _analyze(client, text=SENSATIONAL_TEXT))["cached"] is True
    async with make_client(after) as (_, client):
        second = await _analyze(client, text=SENSATIONAL_TEXT)
        third = await _analyze(client, text=SENSATIONAL_TEXT)

    assert first["classifier"]["model"] == "baseline"
    assert second["cached"] is False
    assert second["classifier"]["model"] == "tiny-fixture"
    assert third["cached"] is True


# ---------------------------------------------------------------- satire


async def test_satire_adds_a_tag_and_keeps_the_verdict(
    client: httpx.AsyncClient, make_client: ClientFactory, full_artifacts: Path, tmp_path: Path
) -> None:
    request = {"title": SATIRE_HEADLINE, "text": NEUTRAL_TEXT}
    root = _copy(full_artifacts, tmp_path / "artifacts")
    shutil.rmtree(root / "satire")
    without_scorer = Settings(var_dir=tmp_path / "other", artifacts_dir=root, log_json=False)

    tagged = await _analyze(client, **request)
    async with make_client(without_scorer) as (_, other):
        plain = await _analyze(other, **request)

    assert tagged["classifier"]["satire_score"] >= 0.80
    assert "satire" in tagged["tags"]
    assert "The headline reads like satire." in tagged["summary"]
    assert plain["classifier"]["satire_score"] is None
    assert "satire" not in plain["tags"]
    assert "satire" not in plain["summary"]
    assert tagged["verdict"] == plain["verdict"]
    assert tagged["risk_score"] == plain["risk_score"]


async def test_an_ordinary_headline_is_not_tagged_as_satire(client: httpx.AsyncClient) -> None:
    body = await _analyze(client, title="Budget approved after review", text=NEUTRAL_TEXT)

    assert body["classifier"]["satire_score"] < 0.80
    assert "satire" not in body["tags"]


# ---------------------------------------------------------------- the real artifact


@pytest.mark.slow
async def test_real_artifact_answers(make_client: ClientFactory, tmp_path: Path) -> None:
    if resolve(REAL_ARTIFACTS, "transformer") is None:
        pytest.skip("ml/artifacts has no transformer")
    real = Settings(var_dir=tmp_path / "var", artifacts_dir=REAL_ARTIFACTS, log_json=False)

    async with make_client(real) as (_, client):
        models = (await client.get("/api/v1/meta")).json()["models"]
        body = await _analyze(client, text=SENSATIONAL_TEXT)

    assert models["transformer"].startswith("v1-")
    assert body["classifier"]["version"] == models["transformer"]
    assert body["classifier"]["calibration"] == "temperature"
    assert _stage(body, "classifier")["status"] == "ok"
    assert 0.01 <= body["classifier"]["p_unreliable"] <= 0.99
