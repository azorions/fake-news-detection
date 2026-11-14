"""``GET /api/v1/meta``: what this deployment runs and what it accepts."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from fnd_api.api.deps import RegistryDep, SettingsDep

router = APIRouter(tags=["meta"])


class Providers(BaseModel):
    external: bool  # True when any provider that calls an outside service is enabled
    enabled: list[str]


class Limits(BaseModel):
    rate_limit_analyze: str
    rate_limit_read: str
    max_text_chars: int
    max_fetch_bytes: int
    fetch_timeout_s: float


class MetaResponse(BaseModel):
    models: dict[str, str]  # model kind -> active version
    providers: Providers
    limits: Limits


@router.get("/meta")
async def meta(settings: SettingsDep, registry: RegistryDep) -> MetaResponse:
    return MetaResponse(
        models=registry.versions,
        providers=Providers(external=False, enabled=[]),
        limits=Limits(
            rate_limit_analyze=settings.rate_limit_analyze,
            rate_limit_read=settings.rate_limit_read,
            max_text_chars=settings.max_text_chars,
            max_fetch_bytes=settings.max_fetch_bytes,
            fetch_timeout_s=settings.fetch_timeout_s,
        ),
    )
