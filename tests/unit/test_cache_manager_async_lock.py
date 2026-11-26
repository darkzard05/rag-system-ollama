"""Concurrent cache writes must not interleave inside the manager's critical section."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from services.optimization.caching_optimizer import CacheManager


@pytest.mark.asyncio
async def test_concurrent_sets_are_mutually_exclusive() -> None:
    manager = CacheManager(
        enable_memory_cache=True,
        enable_semantic_cache=False,
        enable_disk_cache=False,
    )
    state = {"active": 0, "peak": 0}

    async def slow_set(*args, **kwargs) -> None:
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        await asyncio.sleep(0)
        state["active"] -= 1

    memory = MagicMock()
    memory.set = AsyncMock(side_effect=slow_set)
    manager.memory_cache = memory

    await asyncio.gather(*(manager.set(f"k{i}", i) for i in range(4)))

    assert state["peak"] == 1
