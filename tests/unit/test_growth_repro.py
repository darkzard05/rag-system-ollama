"""GATE-1b growth repro: multi-session state with no purge grows without bound.

Red-phase (TDD) harness for hyperplan T1. Shows monotonic growth in
process-global dicts/registries that have no cap, LRU, TTL sweeper, or
automatic purge:

* ``ResourceCoordinator._build_locks`` — the dict itself has no intrinsic
  bound (no max-size/LRU/TTL). ``_cleanup_build_locks``
  (resource_manager.py:466-487) only runs on call-site cadence (every 50
  builds: lines 641-643, 701-703; plus ``clear_vram``/``clear_all``) and only
  removes keys absent from the pools AND not in-flight — so every cached
  entry keeps its lock and failed/never-built keys linger between sweeps.
* ``ResourceCoordinator._build_failures`` — negative-cache entries are
  removed only on later success (lines 672, 728) or on a TTL-expired *read*
  (line 511). Unique failed keys with no retry accumulate forever.
* ``core.graph._speculative_gen._spec_registry`` — per-thread_id entries are
  popped only on adopt (``_adopt_speculative_generate``) or same-thread
  restart (``_start_speculative_generate``). Abandoned thread_ids (session
  never completes the generate path) are never reaped; there is no sweeper.

One characterization test documents the sufficient sub-case
(``_cleanup_build_locks`` DOES purge idle non-pool keys when it runs).
"""

import asyncio
from collections.abc import Iterator
from typing import Any

import pytest

from core.graph._speculative_gen import (
    _adopt_speculative_generate,
    _spec_registry,
    _SpecGenerate,
)
from core.resource_manager import get_resource_manager

_BUILD_LOCK_KEYS = 100
_BUILD_LOCK_BOUND = 10
_FAILURE_KEYS = 25
_FAILURE_BOUND = 5
_ORPHAN_THREADS = 10
_ORPHAN_BOUND = 1


@pytest.fixture(autouse=True)
def _isolate_growth_globals() -> Iterator[None]:
    """Reset growth surfaces so repros never leak into other tests."""
    get_resource_manager().reset()
    _spec_registry.clear()
    yield
    get_resource_manager().reset()
    _spec_registry.clear()


def test_build_locks_have_no_intrinsic_bound() -> None:
    """GATE-1b: N distinct build keys leave N entries in ``_build_locks``.

    The dict has no max-size/LRU/TTL of its own; boundedness relies entirely
    on the every-50-builds call-site sweep, which additionally spares every
    key present in the pools. Unique keys therefore grow the dict
    monotonically between (and beyond) sweeps.
    """
    coordinator = get_resource_manager()
    for i in range(_BUILD_LOCK_KEYS):
        coordinator._get_build_lock(f"t1-growth-build-{i}")
    size = len(coordinator._build_locks)
    assert size <= _BUILD_LOCK_BOUND, (
        f"_build_locks grew to {size} entries for {_BUILD_LOCK_KEYS} "
        "distinct keys with no intrinsic bound (no cap/LRU/TTL on the dict)"
    )


def test_build_failures_are_never_purged() -> None:
    """GATE-1b: N distinct failed keys leave N entries in ``_build_failures``.

    Failure records are dropped only on a later success for the SAME key or
    on a TTL-expired read of that key — there is no background purge and no
    size cap, so offline-flapping keys (e.g. per-file retriever builds)
    accumulate one entry each, forever.
    """
    coordinator = get_resource_manager()
    for i in range(_FAILURE_KEYS):
        coordinator._record_build_failure(f"t1-growth-failure-{i}")
    size = len(coordinator._build_failures)
    assert size <= _FAILURE_BOUND, (
        f"_build_failures grew to {size} entries for {_FAILURE_KEYS} "
        "distinct failed keys with no purge path (no cap/sweeper)"
    )


async def _never_finishes() -> dict[str, Any]:
    """Placeholder speculative body that stays pending until cancelled."""
    await asyncio.sleep(3600.0)
    return {}


def test_spec_registry_orphans_are_never_reaped() -> None:
    """GATE-1b: abandoned thread_ids stay in ``_spec_registry`` forever.

    Ten speculative tasks are registered on distinct thread_ids (simulating
    ten sessions); only one completes the adopt path. The other nine have no
    reaper — pop happens solely on adopt or same-thread restart — so the
    registry grows monotonically with abandoned sessions.
    """
    previous: asyncio.AbstractEventLoop | None = None
    try:
        previous = asyncio.get_event_loop()
    except RuntimeError:
        previous = None
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks: list[asyncio.Task[dict[str, Any]]] = []
    try:
        for i in range(_ORPHAN_THREADS):
            task = loop.create_task(_never_finishes())
            tasks.append(task)
            _spec_registry[f"t1-growth-thread-{i}"] = _SpecGenerate(
                task=task, buffer=[]
            )
        adopted = _adopt_speculative_generate(
            {"configurable": {"thread_id": "t1-growth-thread-0"}}
        )
        assert adopted is not None
        size = len(_spec_registry)
        assert size <= _ORPHAN_BOUND, (
            f"_spec_registry holds {size} entries after {_ORPHAN_THREADS} "
            "registrations with a single adopt: abandoned thread_ids are "
            "never reaped (no sweeper/TTL)"
        )
    finally:
        for task in tasks:
            task.cancel()
        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        loop.close()
        if previous is not None:
            asyncio.set_event_loop(previous)
        _spec_registry.clear()


def test_cleanup_purges_idle_nonpool_locks_characterization() -> None:
    """Characterize the sufficient sub-case (passes): idle non-pool locks purge.

    When ``_cleanup_build_locks`` DOES run, keys absent from every pool and
    not in-flight are removed. This documents the existing partial
    mitigation with evidence — the gap proven above is everything the sweep
    does NOT cover (pool-resident keys, in-flight keys, cadence gaps,
    ``_build_failures``, ``_spec_registry``).
    """
    coordinator = get_resource_manager()
    for i in range(5):
        coordinator._get_build_lock(f"t1-growth-idle-{i}")
    assert len(coordinator._build_locks) == 5
    coordinator._cleanup_build_locks()
    assert len(coordinator._build_locks) == 0
