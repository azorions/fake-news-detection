"""Request dependencies: the objects the lifespan put on ``app.state``."""

from __future__ import annotations

from typing import Annotated

import httpx
from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncEngine

from fnd_api.cache import Cache
from fnd_api.config import Settings
from fnd_api.db.repositories import FeedbackRepository, ScanRepository
from fnd_api.models_registry import ModelRegistry
from fnd_api.pipeline.context import PipelineContext


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_registry(request: Request) -> ModelRegistry:
    registry: ModelRegistry = request.app.state.models
    return registry


def get_http_client(request: Request) -> httpx.AsyncClient:
    client: httpx.AsyncClient = request.app.state.http
    return client


def get_engine(request: Request) -> AsyncEngine:
    engine: AsyncEngine = request.app.state.engine
    return engine


def get_cache(request: Request) -> Cache:
    cache: Cache = request.app.state.cache
    return cache


def get_scans(request: Request) -> ScanRepository:
    settings: Settings = request.app.state.settings
    return ScanRepository(request.app.state.sessions, settings.store_input_text)


def get_feedback(request: Request) -> FeedbackRepository:
    return FeedbackRepository(request.app.state.sessions)


def get_pipeline_context(request: Request) -> PipelineContext:
    state = request.app.state
    return PipelineContext(
        settings=state.settings,
        models=state.models,
        cache=state.cache,
        http=state.http,
        rules=state.rules,
        scans=get_scans(request),
        resolver=state.resolver,
    )


SettingsDep = Annotated[Settings, Depends(get_settings)]
PipelineDep = Annotated[PipelineContext, Depends(get_pipeline_context)]
RegistryDep = Annotated[ModelRegistry, Depends(get_registry)]
HttpClientDep = Annotated[httpx.AsyncClient, Depends(get_http_client)]
EngineDep = Annotated[AsyncEngine, Depends(get_engine)]
CacheDep = Annotated[Cache, Depends(get_cache)]
ScansDep = Annotated[ScanRepository, Depends(get_scans)]
FeedbackDep = Annotated[FeedbackRepository, Depends(get_feedback)]
