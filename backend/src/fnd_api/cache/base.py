from __future__ import annotations

from typing import Protocol


class Cache(Protocol):
    """A key-value store with per-entry expiry. Values are opaque bytes."""

    async def get(self, key: str) -> bytes | None: ...

    async def set(self, key: str, value: bytes, ttl_s: int) -> None: ...

    async def delete(self, key: str) -> None: ...

    async def close(self) -> None: ...
