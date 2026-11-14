"""Fixtures for API tests: a tiny synthetic baseline artifact and an in-process HTTP client."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from fnd_api.config import Settings
from fnd_api.main import create_app
from fnd_ml.common.artifacts import register
from fnd_ml.models.baseline import BaselineConfig, BaselineModel
from tests.fixtures.synthetic import make_corpus

BASELINE_VERSION = "v-test"


@pytest.fixture(scope="session")
def artifacts_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An artifacts directory whose active baseline is trained on the synthetic corpus."""
    root = tmp_path_factory.mktemp("artifacts")
    frame = make_corpus(240, seed=1)
    texts, y = frame["model_text"].tolist(), frame["label"].to_numpy()
    model = BaselineModel(BaselineConfig(n_features=2**16)).fit(texts, y).calibrate(texts, y)
    directory = root / "baseline" / BASELINE_VERSION
    model.save(directory, {"version": BASELINE_VERSION})
    (directory / "thresholds.json").write_text('{"t_low": 0.35, "t_high": 0.65}', encoding="utf-8")
    register(root, "baseline", BASELINE_VERSION, f"baseline/{BASELINE_VERSION}")
    return root


@pytest.fixture
def settings(tmp_path: Path, artifacts_dir: Path) -> Settings:
    return Settings(var_dir=tmp_path / "var", artifacts_dir=artifacts_dir, log_json=False)


@pytest.fixture
async def app(settings: Settings) -> AsyncIterator[FastAPI]:
    application = create_app(settings)
    async with application.router.lifespan_context(application):
        yield application


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        yield http
