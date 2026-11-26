"""Overwriting the LRU entry must not double-subtract its tracked size."""

import sys

import pytest

from services.optimization.backends import MemoryCache


@pytest.mark.asyncio
async def test_overwriting_lru_keeps_accurate_memory_accounting() -> None:
    cache = MemoryCache(max_size=2)
    old = "A" * 100
    other = "B" * 100
    replacement = "C" * 200

    await cache.set("a", old)
    await cache.set("b", other)
    await cache.set("a", replacement)

    assert await cache.get("a") == replacement
    assert await cache.get("b") == other
    stats = cache.get_stats()
    assert stats.cache_size == 2
    assert stats.total_memory_bytes == sys.getsizeof(other) + sys.getsizeof(replacement)
    assert stats.total_evictions == 0
