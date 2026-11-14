"""Application factory. Run with ``uvicorn fnd_api.main:create_app --factory``."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import anyio.to_thread
import httpx
from fastapi import FastAPI

from fnd_api import __version__
from fnd_api.api.errors import install_error_handlers
from fnd_api.api.v1 import health, meta
from fnd_api.cache import make_cache
from fnd_api.config import Settings
from fnd_api.db.session import init_db, session_factory
from fnd_api.logging import RequestContextMiddleware, configure_logging, get_logger
from fnd_api.models_registry import ModelRegistry

USER_AGENT = f"fnd-api/{__version__}"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    configure_logging(settings.log_json, settings.log_level)
    log = get_logger("fnd_api")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.models = await anyio.to_thread.run_sync(
            ModelRegistry.load, settings.artifacts_dir
        )
        if not app.state.models.ready:
            log.warning("no_model_artifact", artifacts_dir=str(settings.artifacts_dir))
        app.state.engine = await init_db(settings)
        app.state.sessions = session_factory(app.state.engine)
        app.state.cache = make_cache(settings)
        # Redirects are followed by hand so each hop can be checked against the SSRF guard.
        app.state.http = httpx.AsyncClient(
            timeout=settings.fetch_timeout_s,
            follow_redirects=False,
            headers={"User-Agent": USER_AGENT},
        )
        log.info("startup", models=app.state.models.versions)
        try:
            yield
        finally:
            await app.state.http.aclose()
            await app.state.cache.close()
            await app.state.engine.dispose()

    app = FastAPI(title="Fake News Detection API", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    install_error_handlers(app)
    app.add_middleware(RequestContextMiddleware)
    app.include_router(health.router)
    app.include_router(meta.router, prefix="/api/v1")
    return app
