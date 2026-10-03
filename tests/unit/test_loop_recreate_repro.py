"""GATE-1a loop-swap repro: asyncio loop-bound global locks fork on recreate.

Red-phase (TDD) harness for hyperplan T1. Proves with evidence whether the
partial guards at ``src/core/resource_manager.py:458-463``
(``_get_build_lock``) and ``src/core/graph/_graph_core.py:146-153``
(``_GraphCache.get_lock``) are sufficient across event-loop recreation.

Loop-swap shape (matches pytest-asyncio ``mode=auto`` per-test loop swap):
loop A builds/gets a lock and holds it, loop A is retired, loop B asks for
the same key with a timeout guard. A correct global must preserve mutual
exclusion across the swap; recreating a fresh UNLOCKED lock for the same key
while the old one is still held opens a double-build / double-compile window.

All blocking acquires are wrapped in ``asyncio.wait_for`` so a hang fails
loud instead of hanging the suite forever. Sync tests drive two explicit
loops so the swap is deterministic (no dependence on the runner's loop).

Binding note (verified against this interpreter's ``asyncio.Lock`` source):
an UNCONTENDED ``acquire`` takes a fast path that never touches the loop,
so ``_loop`` stays ``None`` until a waiter parks. The guards under test key
off ``_loop``, therefore every repro below first parks a waiter
(``_park_one_waiter``) to force the binding — exactly the contended state a
real in-flight build/compile reaches.
"""

import asyncio
import contextlib
from collections.abc import Iterator
from typing import Any

import pytest

from common.exceptions import LLMInferenceError
from core.graph._graph_core import _graph_cache
from core.resource_manager import get_resource_manager

_ACQUIRE_TIMEOUT_SECONDS = 2.0
_LOOP_B_ACQUIRE_TIMEOUT_SECONDS = 1.0


def _run(coro: Any, loop: asyncio.AbstractEventLoop) -> Any:
    """Drive ``coro`` on ``loop`` with a loud timeout guard (hangs fail)."""
    return loop.run_until_complete(
        asyncio.wait_for(coro, timeout=_ACQUIRE_TIMEOUT_SECONDS)
    )


@pytest.fixture(autouse=True)
def _isolate_async_globals() -> Iterator[None]:
    """Reset loop-bound singletons so repros never leak into other tests."""
    get_resource_manager().reset()
    _graph_cache._lock = None
    yield
    get_resource_manager().reset()
    _graph_cache._lock = None


def _swap_to_fresh_loop_pair() -> tuple[asyncio.AbstractEventLoop, Any]:
    """Create (loop_a, previous_loop) with loop_a installed as current."""
    previous = asyncio.get_event_loop()
    loop_a = asyncio.new_event_loop()
    asyncio.set_event_loop(loop_a)
    return loop_a, previous


async def _park_one_waiter(lock: asyncio.Lock) -> None:
    """Park then cancel one waiter so the contended path binds ``_loop``."""
    waiter = asyncio.ensure_future(lock.acquire())
    await asyncio.sleep(0.05)
    waiter.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await waiter


def _hold_and_bind(lock: asyncio.Lock, loop: asyncio.AbstractEventLoop) -> None:
    """Acquire ``lock`` and force-bind its ``_loop`` to ``loop`` (stays held)."""
    _run(lock.acquire(), loop)
    loop.run_until_complete(
        asyncio.wait_for(_park_one_waiter(lock), timeout=_ACQUIRE_TIMEOUT_SECONDS)
    )
    assert lock.locked()


def test_build_lock_forks_mutual_exclusion_across_loop_recreate() -> None:
    """GATE-1a: same-key build lock held on loop A must stay exclusive on B.

    Expected (fixed): ``_get_build_lock`` on loop B returns the SAME lock
    object (or one that still reports locked) while loop A's holder is
    in-flight. Actual (current code): the guard at resource_manager.py:458-463
    recreates a fresh unlocked lock because ``_loop`` differs, so ``locked()``
    is False on loop B while loop A still holds the critical section.
    """
    coordinator = get_resource_manager()
    key = "t1-loop-fork-build-lock"
    loop_a, previous = _swap_to_fresh_loop_pair()
    loop_b = asyncio.new_event_loop()
    lock_a: asyncio.Lock | None = None
    try:
        lock_a = coordinator._get_build_lock(key)
        assert lock_a is not None
        _hold_and_bind(lock_a, loop_a)

        asyncio.set_event_loop(loop_b)
        lock_b = coordinator._get_build_lock(key)
        assert lock_a is lock_b or lock_b.locked(), (
            "loop swap forked mutual exclusion: loop B got a fresh "
            f"unlocked lock for {key!r} while loop A still holds it "
            "(double-build window)"
        )
    finally:
        if lock_a is not None and lock_a.locked():
            lock_a.release()
        loop_a.close()
        loop_b.close()
        asyncio.set_event_loop(previous)
        coordinator.reset()


def test_graph_cache_lock_forks_while_held_across_loop_recreate() -> None:
    """GATE-1a: graph build lock held on loop A must stay exclusive on B.

    Same fork as above through ``_GraphCache.get_lock``
    (_graph_core.py:146-153): a compile in-flight on loop A is invisible to
    loop B, which gets a fresh unlocked lock and can double-compile.
    """
    loop_a, previous = _swap_to_fresh_loop_pair()
    loop_b = asyncio.new_event_loop()
    lock_a: asyncio.Lock | None = None
    try:
        lock_a = _graph_cache.get_lock()
        assert lock_a is not None
        _hold_and_bind(lock_a, loop_a)

        asyncio.set_event_loop(loop_b)
        lock_b = _graph_cache.get_lock()
        assert lock_a is lock_b or lock_b.locked(), (
            "graph lock forked across loop swap: loop B got a fresh "
            "unlocked build lock while loop A still holds it "
            "(double-compile window)"
        )
    finally:
        if lock_a is not None and lock_a.locked():
            lock_a.release()
        loop_a.close()
        loop_b.close()
        asyncio.set_event_loop(previous)
        _graph_cache._lock = None


def test_stale_lock_acquire_raises_on_foreign_loop_characterization() -> None:
    """Characterize the hazard the recreate-guards protect against (passes).

    A lock bound to loop A (holder + parked waiter forces the binding, then
    both are released) raises ``RuntimeError: ... bound to a different event
    loop`` when a WAITER parks on loop B — the contended path is the only
    one that consults ``_loop``. This is WHY the guards recreate — but
    recreation is exactly what forks mutual exclusion while the lock is HELD
    (see the two tests above).
    """
    loop_a, previous = _swap_to_fresh_loop_pair()
    loop_b = asyncio.new_event_loop()
    stale: asyncio.Lock | None = None
    try:
        stale = asyncio.Lock()
        _hold_and_bind(stale, loop_a)
        stale.release()
        assert stale.locked() is False

        asyncio.set_event_loop(loop_b)
        # Fast-path acquire succeeds (it never consults _loop); the holder
        # forces the waiter below onto the contended path, which raises.
        _run(stale.acquire(), loop_b)
        with pytest.raises(RuntimeError, match="different event loop"):
            loop_b.run_until_complete(
                asyncio.wait_for(
                    _park_one_waiter(stale), timeout=_ACQUIRE_TIMEOUT_SECONDS
                )
            )
    finally:
        if stale is not None and stale.locked():
            stale.release()
        loop_a.close()
        loop_b.close()
        asyncio.set_event_loop(previous)


def test_inference_semaphore_drops_held_permit_on_loop_swap() -> None:
    """GATE-1a: held inference permit (bound=1) must survive a loop swap.

    ``config.yml`` sets ``max_concurrent_inference: 1``. Holding the single
    permit on loop A must make a loop B acquire time out
    (``LLMInferenceError``). Actual: ``acquire_inference_lock``
    (resource_manager.py:556-560) recreates the semaphore on loop change,
    discarding the held-permit accounting, so loop B acquires immediately
    (over-admission: 2 concurrent inferences against a bound of 1).
    """
    coordinator = get_resource_manager()
    loop_a, previous = _swap_to_fresh_loop_pair()
    loop_b = asyncio.new_event_loop()
    acquired_on_b = False
    try:
        _run(
            coordinator.acquire_inference_lock(timeout=_LOOP_B_ACQUIRE_TIMEOUT_SECONDS),
            loop_a,
        )
        assert coordinator._inference_semaphore is not None

        asyncio.set_event_loop(loop_b)
        try:
            loop_b.run_until_complete(
                asyncio.wait_for(
                    coordinator.acquire_inference_lock(timeout=0.5),
                    timeout=_ACQUIRE_TIMEOUT_SECONDS,
                )
            )
            acquired_on_b = True
        except LLMInferenceError:
            acquired_on_b = False
        assert acquired_on_b is False, (
            "loop swap dropped held-permit accounting: loop B acquired the "
            "inference semaphore while loop A still holds the only permit "
            "(bound=1 over-admission)"
        )
    finally:
        if acquired_on_b:
            coordinator.release_inference_lock()
        loop_a.close()
        loop_b.close()
        asyncio.set_event_loop(previous)
        coordinator.reset()
