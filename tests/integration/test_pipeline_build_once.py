"""Regression: build-lock recycling must not cause a double-build.

Migrated from ``scripts/repro_double_build.py``.

Scenario (see ``src/core/resource_manager.py``):
  - ``_get_build_lock(key)`` creates a fresh ``asyncio.Lock`` if key absent.
  - ``get_or_build`` bumps ``_build_call_counter`` and every 50th call runs
    ``_cleanup_build_locks()`` INSIDE the build lock.
  - ``_cleanup_build_locks`` deletes any key not present in the models /
    retrievers pools. During a build, key K is absent from those pools, so it
    is deleted and a concurrent caller gets a NEW lock for K, double-building.

This guard primes the counter to 49 so the first K call is the 50th (triggering
mid-build cleanup), then runs two concurrent ``get_or_build(K)`` calls. The
build must run exactly once.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from core.resource_manager import ResourceCoordinator


class FakePool:
    """Minimal duck-typed pool mirroring ``BaseResourcePool._pool``."""

    def __init__(self) -> None:
        self._pool: dict[str, Any] = {}

    def get(self, key: str) -> Any | None:
        return self._pool.get(key)

    async def put(self, key: str, res: Any) -> None:
        self._pool[key] = res


@pytest.mark.asyncio
async def test_concurrent_build_is_deduplicated() -> None:
    coord = ResourceCoordinator()
    coord.reset()  # deterministic clean state
    pool = FakePool()

    key = "double_build_target"
    build_count = {"n": 0}

    async def slow_build() -> Any:
        build_count["n"] += 1
        await asyncio.sleep(0.3)  # widen the in-flight window (await yields)
        return object()

    # Prime so the first K call is the 50th -> cleanup runs mid-build.
    coord._build_call_counter = 49

    results = await asyncio.gather(
        coord.get_or_build(pool, key, slow_build),
        coord.get_or_build(pool, key, slow_build),
    )

    # The key must be built exactly once despite the cleanup racing the build.
    assert build_count["n"] == 1, f"key '{key}' built {build_count['n']} times"
    # Both callers observe the same cached object.
    assert results[0] is results[1]
