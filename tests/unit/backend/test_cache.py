import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import fakeredis
import pytest

from fnd_api.cache import Cache, make_cache
from fnd_api.cache.disk_cache import DiskCache
from fnd_api.cache.keys import scan_key
from fnd_api.cache.redis_cache import RedisCache
from fnd_api.config import Settings
from fnd_api.schemas import AnalyzeOptions, Depth


@pytest.fixture(params=["disk", "redis"])
async def cache(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[Cache]:
    backend: Cache
    if request.param == "disk":
        backend = DiskCache(tmp_path / "cache")
    else:
        backend = RedisCache(fakeredis.FakeAsyncRedis())
    yield backend
    await backend.close()


# ---------------------------------------------------------------- contract (both backends)


async def test_set_then_get_returns_the_bytes(cache: Cache) -> None:
    assert await cache.get("missing") is None
    await cache.set("k", b"\x00binary\xff", ttl_s=60)
    assert await cache.get("k") == b"\x00binary\xff"


async def test_set_overwrites_and_delete_removes(cache: Cache) -> None:
    await cache.set("k", b"one", ttl_s=60)
    await cache.set("k", b"two", ttl_s=60)
    assert await cache.get("k") == b"two"
    await cache.delete("k")
    await cache.delete("k")  # deleting a missing key is not an error
    assert await cache.get("k") is None


async def test_entries_expire_after_their_ttl(cache: Cache) -> None:
    await cache.set("short", b"v", ttl_s=1)
    await cache.set("long", b"v", ttl_s=60)
    assert await cache.get("short") == b"v"
    await asyncio.sleep(1.2)
    assert await cache.get("short") is None
    assert await cache.get("long") == b"v"


async def test_concurrent_use_is_safe(cache: Cache) -> None:
    await asyncio.gather(*(cache.set(f"k{i}", str(i).encode(), ttl_s=60) for i in range(20)))
    values = await asyncio.gather(*(cache.get(f"k{i}") for i in range(20)))
    assert values == [str(i).encode() for i in range(20)]


# ---------------------------------------------------------------- factory


async def test_make_cache_uses_the_disk_unless_redis_is_configured(tmp_path: Path) -> None:
    on_disk = make_cache(Settings(var_dir=tmp_path))
    assert isinstance(on_disk, DiskCache)
    await on_disk.set("k", b"v", ttl_s=60)
    await on_disk.close()
    assert (tmp_path / "cache").is_dir()

    reopened = make_cache(Settings(var_dir=tmp_path))
    assert await reopened.get("k") == b"v"  # survives a restart
    await reopened.close()

    in_redis = make_cache(Settings(var_dir=tmp_path, redis_url="redis://localhost:6379/0"))
    assert isinstance(in_redis, RedisCache)
    await in_redis.close()


# ---------------------------------------------------------------- keys


def test_scan_key_depends_on_input_depth_and_model_versions() -> None:
    options = AnalyzeOptions()
    models = {"baseline": "v1"}
    key = scan_key("some text", options, models)
    assert key == scan_key("some text", AnalyzeOptions(), {"baseline": "v1"})
    assert key.startswith("scan:")
    assert key != scan_key("other text", options, models)
    assert key != scan_key("some text", AnalyzeOptions(depth=Depth.DEEP), models)
    assert key != scan_key("some text", AnalyzeOptions(max_claims=3), models)
    assert key != scan_key("some text", AnalyzeOptions(allow_external=False), models)
    assert key != scan_key("some text", options, {"baseline": "v2"})
    assert key != scan_key("some text", options, {"baseline": "v1", "transformer": "t1"})


def test_scan_key_ignores_cache_and_storage_switches_and_dict_order() -> None:
    models = {"baseline": "v1", "transformer": "t1"}
    key = scan_key("some text", AnalyzeOptions(), models)
    assert key == scan_key("some text", AnalyzeOptions(use_cache=False, store=False), models)
    assert key == scan_key("some text", AnalyzeOptions(), {"transformer": "t1", "baseline": "v1"})


def test_scan_key_does_not_contain_the_input() -> None:
    assert "secret" not in scan_key("secret text", AnalyzeOptions(), {})
