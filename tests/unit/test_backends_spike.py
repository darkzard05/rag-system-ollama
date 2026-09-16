"""Spike test for _backends extraction — baseline for Batch 5.1."""

import time

import pytest

from services.optimization._backends import CacheEntry, MemoryCache


def test_cache_entry_creation() -> None:
    now = time.time()
    entry = CacheEntry(
        key="k1",
        value={"hello": "world"},
        created_at=now,
        accessed_at=now,
        ttl_seconds=60.0,
    )
    assert entry.key == "k1"
    assert entry.value == {"hello": "world"}
    assert entry.is_expired() is False
    assert entry.get_age() >= 0
    entry.touch()
    assert entry.hit_count == 1


def test_cache_entry_expiry() -> None:
    now = time.time()
    entry = CacheEntry(
        key="k2",
        value="v",
        created_at=now - 10,
        accessed_at=now - 10,
        ttl_seconds=5.0,
    )
    assert entry.is_expired() is True

    entry_never = CacheEntry(
        key="k3",
        value="v",
        created_at=now - 1000,
        accessed_at=now,
        ttl_seconds=0,
    )
    assert entry_never.is_expired() is False


def test_cache_entry_json_roundtrip() -> None:
    now = time.time()
    entry = CacheEntry(
        key="jk",
        value="json-val",
        created_at=now,
        accessed_at=now,
        ttl_seconds=30.0,
        hit_count=2,
        metadata={"x": 1},
    )
    d = entry.to_json_dict()
    restored = CacheEntry.from_json_dict(d)
    assert restored.key == entry.key
    assert restored.value == entry.value
    assert restored.ttl_seconds == entry.ttl_seconds
    assert restored.hit_count == entry.hit_count
    assert restored.metadata == entry.metadata


@pytest.mark.asyncio
async def test_memory_cache_set_get_roundtrip() -> None:
    cache: MemoryCache[str] = MemoryCache(max_size=10, max_memory_mb=10, ttl_seconds=60)
    assert await cache.get("missing") is None

    await cache.set("hello", "world")
    val = await cache.get("hello")
    assert val == "world"

    # overwrite
    await cache.set("hello", "again")
    assert await cache.get("hello") == "again"

    # delete
    await cache.delete("hello")
    assert await cache.get("hello") is None

    # clear
    await cache.set("a", "1")
    await cache.set("b", "2")
    await cache.clear()
    assert await cache.get("a") is None
    assert await cache.get("b") is None

    stats = cache.get_stats()
    assert stats.cache_size == 0
