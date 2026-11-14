"""Cache in Redis, for deployments with more than one API process."""

from __future__ import annotations

from typing import Self

from redis.asyncio import Redis

KEY_PREFIX = "fnd:"


class RedisCache:
    def __init__(self, client: Redis) -> None:
        self._client = client

    @classmethod
    def from_url(cls, url: str) -> Self:
        return cls(Redis.from_url(url))

    async def get(self, key: str) -> bytes | None:
        value = await self._client.get(KEY_PREFIX + key)
        return value if isinstance(value, bytes) else None

    async def set(self, key: str, value: bytes, ttl_s: int) -> None:
        await self._client.set(KEY_PREFIX + key, value, ex=ttl_s)

    async def delete(self, key: str) -> None:
        await self._client.delete(KEY_PREFIX + key)

    async def close(self) -> None:
        await self._client.aclose()
