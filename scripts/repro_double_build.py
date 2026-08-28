"""Reproduction for build-lock recycling double-build defect in resource_manager.

Scenario (see src/core/resource_manager.py):
  - _get_build_lock(key) creates a fresh asyncio.Lock if key absent (L353-356).
  - get_or_build bumps _build_call_counter and every 50th call runs
    _cleanup_build_locks() INSIDE the build lock (L518-523).
  - _cleanup_build_locks deletes any key not present in models/retrievers pools
    (L358-367). During a build, key K is absent from those pools, so it is deleted.
  - A concurrent caller then gets a NEW lock for K and double-builds.

This harness primes the counter to 49, launches two concurrent get_or_build(K)
calls where the build_fn sleeps to widen the in-flight window. If the defect
exists, slow_build runs more than once for K.

Exits 2 when DEFECT CONFIRMED, 0 when safe.
"""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT))

from core.resource_manager import ResourceCoordinator


class FakePool:
    """Minimal duck-typed pool: get/put mirroring BaseResourcePool._pool."""

    def __init__(self) -> None:
        self._pool: dict[str, object] = {}

    def get(self, key: str) -> object | None:
        return self._pool.get(key)

    async def put(self, key: str, res: object) -> None:
        self._pool[key] = res


async def main() -> int:
    coord = ResourceCoordinator()
    coord.reset()  # deterministic clean state
    pool = FakePool()

    KEY = "double_build_target"
    build_count = {"n": 0}

    async def slow_build() -> object:
        build_count["n"] += 1
        await asyncio.sleep(0.3)  # simulate heavy model load (await yields)
        return object()

    # Prime so the FIRST K call is the 50th -> triggers cleanup mid-build,
    # deleting the in-flight key from _build_locks.
    coord._build_call_counter = 49

    results = await asyncio.gather(
        coord.get_or_build(pool, KEY, slow_build),
        coord.get_or_build(pool, KEY, slow_build),
    )

    print(f"build_count for key={KEY}: {build_count['n']}")
    print(f"returned objects identical: {results[0] is results[1]}")
    if build_count["n"] > 1:
        print(
            f"DEFECT CONFIRMED: key '{KEY}' built {build_count['n']} times under concurrency"
        )
        return 2
    print("OK: built exactly once")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
