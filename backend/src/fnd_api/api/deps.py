"""Request dependencies: the objects the lifespan put on ``app.state``."""

from __future__ import annotations

from typing import Annotated

import httpx
from fastapi import Depends, Request

from fnd_api.config import Settings
from fnd_api.models_registry import ModelRegistry


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_registry(request: Request) -> ModelRegistry:
    registry: ModelRegistry = request.app.state.models
    return registry


def get_http_client(request: Request) -> httpx.AsyncClient:
    client: httpx.AsyncClient = request.app.state.http
    return client


SettingsDep = Annotated[Settings, Depends(get_settings)]
RegistryDep = Annotated[ModelRegistry, Depends(get_registry)]
HttpClientDep = Annotated[httpx.AsyncClient, Depends(get_http_client)]
