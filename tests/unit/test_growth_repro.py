"""GATE-1b growth characterization: process-global surfaces are bounded.

T1 fix (``f343ba2``) moved each previously unbounded growth surface onto a
write-path cap, so unique keys can no longer grow it without bound:

* ``ResourceCoordinator._build_locks`` — capped at ``_BUILD_LOCK_MAX_ENTRIES``
  by ``_evict_excess_build_locks``, which evicts idle, non-in-flight keys on
  every insert (held/in-flight keys are spared for correctness). The
  call-site ``_cleanup_build_locks`` sweep additionally reaps keys absent from
  every pool.
* ``ResourceCoordinator._build_failures`` — capped at
  ``_BUILD_FAILURE_MAX_ENTRIES`` by ``_record_build_failure``, which first
  purges TTL-expired records then evicts the oldest beyond the cap.
* ``core.graph._speculative_gen._spec_registry`` — one entry per thread_id,
  capped at ``_SPEC_REGISTRY_MAX_SIZE`` by ``_enforce_spec_registry_bound`` on
  the registration write path.

Each surface is bounded on its write path (the cap is enforced wherever the
entry is inserted), which is what the tests below assert.

One characterization test documents the sufficient sub-case
(``_cleanup_build_locks`` DOES purge idle non-pool keys when it runs).
"""

import asyncio
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest

from core.graph._speculative_gen import (
    _SPEC_REGISTRY_MAX_SIZE,
    _spec_registry,
)
from core.resource_manager import (
    _BUILD_FAILURE_MAX_ENTRIES,
    _BUILD_LOCK_MAX_ENTRIES,
    get_resource_manager,
)

_BUILD_LOCK_KEYS = _BUILD_LOCK_MAX_ENTRIES * 10
_FAILURE_KEYS = _BUILD_FAILURE_MAX_ENTRIES + 20
_SPEC_THREADS = _SPEC_REGISTRY_MAX_SIZE + 20


@pytest.fixture(autouse=True)
def _isolate_growth_globals() -> Iterator[None]:
    """Reset growth surfaces so tests never leak into each other."""
    get_resource_manager().reset()
    _spec_registry.clear()
    yield
    get_resource_manager().reset()
    _spec_registry.clear()


def test_build_locks_are_bounded_on_write_path() -> None:
    """GATE-1b: N distinct build keys are capped at ``_BUILD_LOCK_MAX_ENTRIES``.

    ``_get_build_lock`` enforces the intrinsic cap on every insert via
    ``_evict_excess_build_locks`` (sparing held/in-flight keys), so unique
    keys cannot grow the dict without bound even between the call-site sweeps.
    """
    coordinator = get_resource_manager()
    for i in range(_BUILD_LOCK_KEYS):
        coordinator._get_build_lock(f"t1-growth-build-{i}")
    size = len(coordinator._build_locks)
    assert size <= _BUILD_LOCK_MAX_ENTRIES, (
        f"_build_locks grew to {size} entries for {_BUILD_LOCK_KEYS} "
        f"distinct keys (cap is {_BUILD_LOCK_MAX_ENTRIES} on the write path)"
    )


def test_build_failures_are_capped_on_write_path() -> None:
    """GATE-1b: N distinct failed keys are capped at ``_BUILD_FAILURE_MAX_ENTRIES``.

    ``_record_build_failure`` purges TTL-expired records and then evicts the
    oldest beyond the cap, so offline-flapping keys (e.g. per-file retriever
    builds) no longer accumulate one entry each forever.
    """
    coordinator = get_resource_manager()
    for i in range(_FAILURE_KEYS):
        coordinator._record_build_failure(f"t1-growth-failure-{i}")
    size = len(coordinator._build_failures)
    assert size <= _BUILD_FAILURE_MAX_ENTRIES, (
        f"_build_failures grew to {size} entries for {_FAILURE_KEYS} "
        f"distinct failed keys (cap is {_BUILD_FAILURE_MAX_ENTRIES} on the write path)"
    )


async def _pending_generate(
    state: dict[str, Any], config: dict[str, Any], *, writer: Any
) -> dict[str, Any]:
    """Placeholder speculative body that stays pending until cancelled."""
    await asyncio.sleep(3600.0)
    return {}


async def test_spec_registry_is_bounded_on_write_path() -> None:
    """GATE-1b: the registry cap is enforced on the write path.

    ``_spec_registry`` holds at most one entry per thread_id and is bounded by
    ``_SPEC_REGISTRY_MAX_SIZE``. Registering far more distinct thread_ids via
    ``_start_speculative_generate`` must reap the oldest entries instead of
    growing without bound — the cap lives inside the registration entrypoint,
    mirroring ``_get_build_lock`` / ``_record_build_failure``.
    """
    from core.graph._graph_core import _start_speculative_generate

    with (
        patch("core.graph._speculative_gen.MAX_CONCURRENT_INFERENCE", 2),
        patch("core.graph._generate.generate", side_effect=_pending_generate),
    ):
        for i in range(_SPEC_THREADS):
            _start_speculative_generate(
                {"input": "q", "relevant_docs": [], "is_cached": False},
                {"configurable": {"thread_id": f"t1-growth-spec-{i}"}},
                None,
            )
        size = len(_spec_registry)
        assert size <= _SPEC_REGISTRY_MAX_SIZE, (
            f"_spec_registry grew to {size} entries for {_SPEC_THREADS} distinct "
            "thread_ids: the write path did not enforce _SPEC_REGISTRY_MAX_SIZE"
        )
        assert size == _SPEC_REGISTRY_MAX_SIZE

    # Cancel surviving pending tasks so no task leaks across tests.
    survivors = [spec.task for spec in list(_spec_registry.values())]
    for task in survivors:
        if not task.done():
            task.cancel()
    await asyncio.gather(*survivors, return_exceptions=True)


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
