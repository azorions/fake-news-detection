"""Scan history: list, read and delete stored analyses."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Response

from fnd_api.api.deps import CacheDep, ScansDep
from fnd_api.api.errors import ApiError
from fnd_api.cache import Cache
from fnd_api.logging import get_logger
from fnd_api.pipeline.orchestrator import SCAN_REF_PREFIX
from fnd_api.ratelimit import limit_read
from fnd_api.schemas import AnalysisResponse, ScanPage, Verdict

router = APIRouter(tags=["history"], dependencies=[Depends(limit_read)])
log = get_logger("fnd_api.scans")


def scan_not_found() -> ApiError:
    return ApiError("not_found", 404, "There is no scan with this id.")


async def _purge_cached_result(cache: Cache, scan_id: str) -> None:
    """Drop the cached result of a deleted scan, so its text does not outlive it."""
    try:
        key = await cache.get(SCAN_REF_PREFIX + scan_id)
        if key:
            await cache.delete(key.decode())
            await cache.delete(SCAN_REF_PREFIX + scan_id)
    except Exception:  # the entry expires on its own; deletion from history already succeeded
        log.warning("cache_purge_failed", scan_id=scan_id, exc_info=True)


@router.get("/scans")
async def list_scans(
    scans: ScansDep,
    cursor: Annotated[str | None, Query(max_length=200)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    verdict: Verdict | None = None,
    domain: Annotated[str | None, Query(max_length=253)] = None,
    q: Annotated[str | None, Query(max_length=200)] = None,
) -> ScanPage:
    try:
        items, next_cursor = await scans.list(cursor, limit, verdict, domain, q)
    except ValueError as error:
        raise ApiError("invalid_input", 422, "The cursor is not valid.") from error
    return ScanPage(items=items, next_cursor=next_cursor)


@router.get("/scans/{scan_id}")
async def get_scan(scan_id: str, scans: ScansDep) -> AnalysisResponse:
    stored = await scans.get(scan_id)
    if stored is None:
        raise scan_not_found()
    return stored


@router.delete("/scans/{scan_id}", status_code=204)
async def delete_scan(scan_id: str, scans: ScansDep, cache: CacheDep) -> Response:
    if not await scans.delete(scan_id):
        raise scan_not_found()
    await _purge_cached_result(cache, scan_id)
    return Response(status_code=204)
