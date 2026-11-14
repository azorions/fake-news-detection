"""Liveness and readiness probes."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from fnd_api.api.deps import RegistryDep

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(registry: RegistryDep) -> JSONResponse:
    checks = {"models": registry.ready}
    ready = all(checks.values())
    body = {"status": "ready" if ready else "not_ready", "checks": checks}
    return JSONResponse(body, status_code=200 if ready else 503)
