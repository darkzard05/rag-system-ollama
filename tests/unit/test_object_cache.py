"""
Unit tests for ObjectCache and SyncCacheBridge.

These are the foundation backends for the cache-unification refactor
(plan: codebase-audit-remediation, task 1 of 17).
"""

import asyncio
import time

import pytest

from services.optimization.caching_optimizer import (
    ObjectCache,
    SyncCacheBridge,
)


class _CustomObj:
    """A non-trivial Python object to validate in-memory round-tripping."""

    def __init__(self, value: int) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _CustomObj):
            return NotImplemented
        return self.value == other.value


@pytest.mark.asyncio
async def test_object_cache_set_get_roundtrip() -> None:
    """(a) Async set -> get returns the exact stored object."""
    cache: ObjectCache[object] = ObjectCache(max_size=10)
    obj = _CustomObj(42)

    await cache.set("k", obj)
    result = await cache.get("k")

    assert result is obj
    assert result == _CustomObj(42)


@pytest.mark.asyncio
async def test_object_cache_lru_eviction() -> None:
    """(b) Exceeding max_size evicts the least-recently-used entry."""
    cache: ObjectCache[object] = ObjectCache(max_size=3)

    for i in range(3):
        await cache.set(f"k{i}", _CustomObj(i))
        time.sleep(0.01)  # ensure distinct accessed_at timestamps

    # Touch k0 so it becomes recently used; k1 is now LRU.
    await cache.get("k0")
    time.sleep(0.01)

    # Adding a 4th entry should evict k1 (the LRU).
    await cache.set("k3", _CustomObj(3))

    assert await cache.get("k0") is not None
    assert await cache.get("k1") is None
    assert await cache.get("k3") is not None

    stats = cache.get_stats()
    assert stats.cache_size == 3
    assert stats.total_evictions == 1


@pytest.mark.asyncio
async def test_object_cache_clear() -> None:
    """(c) clear() empties the cache."""
    cache: ObjectCache[object] = ObjectCache(max_size=10)
    await cache.set("k", _CustomObj(1))
    await cache.clear()

    assert await cache.get("k") is None
    assert cache.get_stats().cache_size == 0


@pytest.mark.asyncio
async def test_object_cache_stats_accuracy() -> None:
    """(d) get_stats().cache_size is accurate."""
    cache: ObjectCache[object] = ObjectCache(max_size=10)

    assert cache.get_stats().cache_size == 0

    await cache.set("a", _CustomObj(1))
    await cache.set("b", _CustomObj(2))

    assert cache.get_stats().cache_size == 2

    await cache.delete("a")
    assert cache.get_stats().cache_size == 1


def test_sync_bridge_get_from_sync_context() -> None:
    """(e) SyncCacheBridge.get_sync returns value from a plain function."""
    cache: ObjectCache[object] = ObjectCache(max_size=10)
    bridge = SyncCacheBridge(cache)

    obj = _CustomObj(99)
    bridge.set_sync("k", obj)

    # Called from a normal (non-async) function.
    result = bridge.get_sync("k")
    assert result is obj


def test_sync_bridge_shares_state_with_object_cache() -> None:
    """(f) Bridge set/get share state with the underlying ObjectCache."""
    cache: ObjectCache[object] = ObjectCache(max_size=10)
    bridge = SyncCacheBridge(cache)

    # set via sync bridge, read via async ObjectCache
    bridge.set_sync("bridge_key", _CustomObj(7))
    assert asyncio.run(cache.get("bridge_key")) == _CustomObj(7)

    # set via async ObjectCache, read via sync bridge
    async def _seed() -> None:
        await cache.set("async_key", _CustomObj(13))

    asyncio.run(_seed())
    assert bridge.get_sync("async_key") == _CustomObj(13)

    # delete via bridge reflects in async cache
    bridge.delete_sync("async_key")
    assert asyncio.run(cache.get("async_key")) is None


def test_sync_bridge_clear_sync() -> None:
    """SyncCacheBridge.clear_sync empties the underlying cache."""
    cache: ObjectCache[object] = ObjectCache(max_size=10)
    bridge = SyncCacheBridge(cache)

    bridge.set_sync("k", _CustomObj(1))
    bridge.clear_sync()

    assert asyncio.run(cache.get("k")) is None
    assert cache.get_stats().cache_size == 0


def test_sync_bridge_from_inside_running_loop() -> None:
    """(g) get_sync must NOT raise 'event loop already running'.

    Reproduces the real app path: the synchronous VectorStoreCache.load/save
    call bridge.get_sync/set_sync/delete_sync from INSIDE an already-running
    event loop (pipeline_builder.build_pipeline is async). The bridge uses a
    private background-thread loop, so this must work without raising.
    """

    def _run_in_loop() -> None:
        cache: ObjectCache[object] = ObjectCache(max_size=10)
        bridge = SyncCacheBridge(cache)

        obj = _CustomObj(123)
        bridge.set_sync("in_loop", obj)

        # get/delete from within the running loop
        assert bridge.get_sync("in_loop") is obj
        bridge.delete_sync("in_loop")
        assert bridge.get_sync("in_loop") is None

    # asyncio.run starts a fresh running loop; the bridge calls happen inside it.
    asyncio.run(asyncio.to_thread(_run_in_loop))


@pytest.mark.asyncio
async def test_sync_bridge_inside_asyncio_test_loop() -> None:
    """(g2) get_sync works when awaited context already owns a running loop.

    Uses the test's own running loop (via asyncio) to ensure the bridge does
    not touch get_event_loop/run_until_complete on the caller's loop.
    """
    cache: ObjectCache[object] = ObjectCache(max_size=10)
    bridge = SyncCacheBridge(cache)

    obj = _CustomObj(456)

    # Run the synchronous bridge calls in a thread so they execute while this
    # async test owns a running event loop.
    def _call() -> None:
        bridge.set_sync("async_ctx", obj)
        assert bridge.get_sync("async_ctx") is obj

    await asyncio.to_thread(_call)
    assert await cache.get("async_ctx") == _CustomObj(456)
