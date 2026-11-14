"""Cache in a local directory (SQLite-backed ``diskcache``). The default; needs no server."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import diskcache


class DiskCache:
    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        # One worker thread owns the store, so its SQLite connection is opened and closed in the
        # same thread and calls never block the event loop.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fnd-cache")
        self._cache = self._executor.submit(diskcache.Cache, str(directory)).result()

    async def _run[T](self, function: Callable[[], T]) -> T:
        return await asyncio.get_running_loop().run_in_executor(self._executor, function)

    async def get(self, key: str) -> bytes | None:
        value = await self._run(lambda: self._cache.get(key))
        return value if isinstance(value, bytes) else None

    async def set(self, key: str, value: bytes, ttl_s: int) -> None:
        await self._run(lambda: self._cache.set(key, value, expire=ttl_s))

    async def delete(self, key: str) -> None:
        await self._run(lambda: self._cache.delete(key))

    async def close(self) -> None:
        await self._run(self._cache.close)
        self._executor.shutdown(wait=True)
