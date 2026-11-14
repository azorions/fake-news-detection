"""Per-client request limits.

Counters live in the memory of this process and belong to one application instance, so the limit
is per process. Clients are told apart by their network address; behind a reverse proxy, run the
server with proxy headers enabled so that address is the real client's.
"""

from __future__ import annotations

import math
import time

from fastapi import Request
from limits import parse
from limits.aio.storage import MemoryStorage
from limits.aio.strategies import MovingWindowRateLimiter

from fnd_api.api.errors import ApiError


class RateLimiter:
    def __init__(self) -> None:
        self._limiter = MovingWindowRateLimiter(MemoryStorage())

    async def check(self, limit: str, scope: str, client: str) -> None:
        """Count one request; raise ``ApiError`` (429) when ``client`` is over ``limit``."""
        item = parse(limit)
        if await self._limiter.hit(item, scope, client):
            return
        stats = await self._limiter.get_window_stats(item, scope, client)
        retry_after = max(1, math.ceil(stats.reset_time - time.time()))
        raise ApiError(
            "rate_limited",
            429,
            f"Too many requests. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )


def _client(request: Request) -> str:
    return request.client.host if request.client else "unknown"


async def limit_analyze(request: Request) -> None:
    state = request.app.state
    await state.limiter.check(state.settings.rate_limit_analyze, "analyze", _client(request))


async def limit_read(request: Request) -> None:
    state = request.app.state
    await state.limiter.check(state.settings.rate_limit_read, "read", _client(request))
