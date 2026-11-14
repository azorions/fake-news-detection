from __future__ import annotations

from fnd_api.cache.base import Cache
from fnd_api.cache.disk_cache import DiskCache
from fnd_api.cache.redis_cache import RedisCache
from fnd_api.config import Settings

__all__ = ["Cache", "DiskCache", "RedisCache", "make_cache"]


def make_cache(settings: Settings) -> Cache:
    """Redis when ``redis_url`` is set, otherwise a directory on disk."""
    if settings.redis_url:
        return RedisCache.from_url(settings.redis_url)
    assert settings.cache_dir is not None  # Settings derives it from var_dir
    return DiskCache(settings.cache_dir)
