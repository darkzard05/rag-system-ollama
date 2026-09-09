"""Step 2.4 tests: hard-cancel + bounded stream-slot gating.

Spec: ``.omo/plans/top3-fixes.md`` Step 2.4 (lines 184-202). The implementation
under test is already live (``src/common/stream_worker.py`` and the
``stream_chunks`` refactor in ``src/ui/components/streaming.py``); this file
only adds regression coverage and never modifies ``src/``.

Levels each test targets:
- test_cancel_stops_llm_within_timeout   -> task level: cancel_stream interrupts a
                                            blocked mock-LM generator and the slot
                                            is released (the plan-sanctioned
                                            deterministic alternative to the
                                            E2E timeout; see also below)
- test_cancel_releases_semaphore         -> stream_worker contract level:
                                            acquire/register/cancel/release with a
                                            real task on a real event loop
- test_bounded_active_stream_slots       -> stream_worker contract level: 2-slot
                                            pool bounds 5 workers, with turnover
- test_stream_chunks_third_blocks_until_slot_free
                                           -> E2E sanity: 2 concurrent
                                            stream_chunks run while a 3rd blocks
                                            until a slot frees; streams finish
                                            naturally (no cancellation involved)
- test_ui_stream_workers_env_override    -> ``common.config`` provenance (Step 2.1)
"""

import asyncio
import contextlib
import importlib
import threading
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Generator
from typing import Any

import pytest

import common.config as config
from common import stream_worker
from core.rag_core import RAGSystem
from ui.components.streaming import stream_chunks


def _wait_until(condition: Callable[[], bool], timeout: float, message: str) -> None:
    """Poll ``condition`` every 20ms until it holds or ``timeout`` elapses.

    Time-bounded polling keeps the assertions deterministic on slow CI-hosted
    threads instead of relying on fixed sleeps.
    """
    deadline = time.monotonic() + timeout
    while True:
        if condition():
            return
        if time.monotonic() >= deadline:
            pytest.fail(message)
        time.sleep(0.02)


def _reset_stream_worker_state() -> None:
    """Drop any previous test's stream_worker globals (plan Step 2.4 fixture).

    A fresh ``_stream_sem`` is rebuilt lazily by ``_stream_semaphore()`` on the
    next acquire, so the bound is picked up from the config module object at
    that moment (never a stale frozen binding). Any semaphore that a leaked
    thread still holds is discarded together with the dropped semaphore object.
    """
    with stream_worker._stream_lock:
        stream_worker._active_slots = 0
        stream_worker._waiting = 0
        stream_worker._active_streams.clear()
    with stream_worker._sem_lock:
        stream_worker._stream_sem = None
        stream_worker._last_limit = 0


@pytest.fixture(autouse=True)
def _clean_stream_worker_state():
    """Isolate stream_worker slot counters + semaphore between tests."""
    _reset_stream_worker_state()
    yield
    _reset_stream_worker_state()


def test_cancel_stops_llm_within_timeout():
    """Task level: cancel_stream() interrupts a blocked mock-LM async generator.

    Level: stream_worker contract over a real task/loop in a background thread
    (the plan's explicitly-sanctioned alternative for a "deterministic and fast"
    unit test; end-to-end timeouts are covered by test_stream_chunks_* below).
    The background loop is driven with ``run_forever`` so it stays alive across
    the cancellation - this keeps ``cancel_stream`` on its ``is_running()`` path
    and avoids the Windows Proactor double-run shutdown race.
    """
    assert stream_worker.acquire_stream_slot(timeout=1.0) is True
    assert stream_worker.active_stream_count() == 1

    run_id = f"stream-worker-cancel-{uuid.uuid4().hex[:8]}"
    loop = asyncio.new_event_loop()
    started = threading.Event()
    gen_closed = threading.Event()
    consumer_done = threading.Event()

    async def mock_lm() -> AsyncGenerator[tuple[str, dict[str, Any]], None]:
        """One status event, then an LLM stalled on its next token."""
        started.set()
        try:
            yield ("custom", {"status": "thinking"})
            await asyncio.sleep(600)  # never completes on its own
        finally:
            # aclose/GeneratorExit (or the propagation of the task cancel) lands
            # here; this is the "mock generator received aclose" assertion.
            gen_closed.set()

    async def consumer_task() -> None:
        """Mirrors stream_chunks' run(): consume the first chunk, then block."""
        stream: AsyncGenerator[tuple[str, dict[str, Any]], None] = mock_lm()
        first = await anext(stream)  # consume the first chunk
        assert first == ("custom", {"status": "thinking"})
        try:
            await anext(stream)  # blocked inside the mock-lm forever
        except asyncio.CancelledError:
            raise
        finally:
            with contextlib.suppress(Exception):
                await stream.aclose()  # mirrors run()'s event_stream.aclose()
            stream_worker.unregister_stream(run_id)
            stream_worker.release_stream_slot()  # mirrors stream_chunks' finally
            consumer_done.set()

    def run_forever_bg() -> None:
        asyncio.set_event_loop(loop)
        task = loop.create_task(consumer_task())
        stream_worker.register_stream(run_id, loop, task)
        loop.run_forever()  # returns only when the test stops the loop

    bg = threading.Thread(target=run_forever_bg, daemon=True)
    try:
        bg.start()
        assert started.wait(5.0)
        stream_worker.cancel_stream(run_id)  # fire-and-forget, like stream_chunks

        t0 = time.monotonic()
        # The consumer drains within seconds of the cancel: its CancelledError
        # handling closes the generator and the slot is released.
        assert consumer_done.wait(5.0), (
            f"stream must complete within 5s, took {time.monotonic() - t0:.2f}s"
        )
        assert gen_closed.wait(5.0), (
            "mock async generator must receive aclose/GeneratorExit (finally ran)"
        )
        elapsed = time.monotonic() - t0
        assert elapsed < 5.0
        _wait_until(
            lambda: stream_worker.active_stream_count() == 0,
            5.0,
            "cancelled stream must release its slot",
        )
        # A subsequent stream starts immediately: the semaphore is available.
        assert stream_worker.acquire_stream_slot(timeout=0.5) is True
        stream_worker.release_stream_slot()
    finally:
        loop.call_soon_threadsafe(loop.stop)
        bg.join(timeout=5.0)
        loop.close()


class _SlotProbe:
    """Shared stream-entry/exit bookkeeping for the concurrent bound tests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.entered: list[int] = []
        self.closed: list[int] = []

    def entered_count(self) -> int:
        with self._lock:
            return len(self.entered)

    def record_entered(self, idx: int) -> None:
        with self._lock:
            self.entered.append(idx)

    def record_closed(self, idx: int) -> None:
        with self._lock:
            self.closed.append(idx)


def _finite_astream_factory(probe: _SlotProbe) -> Callable[..., Any]:
    """Mock RAGSystem.astream: each stream yields 2 status events, holds its slot
    for 0.8s, then finishes NATURALLY (no cancellation involved anywhere)."""

    async def hanging_astream(
        self: Any, query: str, model_name: str | None = None
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        async def _stream() -> AsyncIterator[tuple[str, dict[str, Any]]]:
            idx = int(query[1:])  # query is "q0".."q2"
            probe.record_entered(idx)
            try:
                yield ("custom", {"status": f"{idx}-1"})
                # Hold the slot briefly (staggered so completions don't overlap),
                # then finish naturally - no cancellation anywhere in this test.
                await asyncio.sleep(0.8 + 0.4 * idx)
                yield ("custom", {"status": f"{idx}-2"})
            finally:
                probe.record_closed(idx)

        return _stream()  # mirrors RAGSystem.astream: returns an async generator

    return hanging_astream


def _consume_stream(chunks: Generator[Any, None, None]) -> None:
    """Drain one stream_chunks generator to its natural end (StopIteration)."""
    try:
        while True:
            next(chunks)
    except StopIteration:
        return


def test_bounded_active_stream_slots(monkeypatch):
    """Contract: a 2-slot pool bounds 5 concurrent workers and admits the next
    waiter when a held slot is released (the way a cancelled stream's
    consumer-side finally releases it).

    Level: stream_worker contract (acquire/release with threads). This is the
    plan-sanctioned deterministic counterpart of the end-to-end bound; the
    end-to-end version lives in test_stream_chunks_third_blocks_until_slot_free.
    """
    # Module-attr patch, NOT env: the lazy accessor reads config.UI_STREAM_WORKERS
    # at call time, so the first acquire below rebuilds a 2-slot semaphore.
    monkeypatch.setattr(config, "UI_STREAM_WORKERS", 2)

    entered_lock = threading.Lock()
    entered: list[int] = []

    def entered_count() -> int:
        with entered_lock:
            return len(entered)

    def _acquire_and_hold(idx: int, release: threading.Event) -> None:
        assert stream_worker.acquire_stream_slot(timeout=30.0) is True
        with entered_lock:
            entered.append(idx)
        release.wait(timeout=60.0)  # "stream is live" until released/cancelled
        stream_worker.release_stream_slot()

    releases = [threading.Event() for _ in range(5)]
    threads: list[threading.Thread] = []
    for i in range(5):
        t = threading.Thread(
            target=_acquire_and_hold,
            args=(i, releases[i]),
            daemon=True,
            name=f"slot-worker-{i}",
        )
        t.start()
        threads.append(t)

    stop_sampling = threading.Event()
    max_active: list[int] = [0]

    def _sample_active() -> None:
        while not stop_sampling.is_set():
            cur = stream_worker.active_stream_count()
            if cur > max_active[0]:
                max_active[0] = cur
            time.sleep(0.005)

    sampler = threading.Thread(target=_sample_active, daemon=True)
    sampler.start()

    def _poll_new_entrant(prev_count: int) -> None:
        _wait_until(
            lambda: entered_count() > prev_count,
            5.0,
            "releasing a held slot should admit a waiting worker",
        )

    try:
        # Assert 2: exactly 2 workers acquired (and hold) slots while the
        # remaining 3 threads are blocked in acquire_stream_slot (waiting == 3).
        _wait_until(
            lambda: entered_count() == 2 and stream_worker.waiting_stream_count() == 3,
            10.0,
            "expect exactly 2 active slots and 3 blocked waiters",
        )
        assert stream_worker.active_stream_count() == 2

        # Assert 3 (slot turnover): release the slot of the first worker (what
        # happens when a cancelled stream's consumer-side finally releases it);
        # within ~1s a different worker acquires -> active returns to 2.
        with entered_lock:
            victim = entered[0]
        released_set = {victim}
        releases[victim].set()
        _wait_until(
            lambda: stream_worker.active_stream_count() == 2 and entered_count() == 3,
            5.0,
            "released slot must be taken over by a waiting worker",
        )

        # Drain: release one held slot at a time until all 5 have entered.
        while entered_count() < 5:
            with entered_lock:
                next_victim = next(i for i in entered if i not in released_set)
            released_set.add(next_victim)
            prev_count = entered_count()
            releases[next_victim].set()
            _poll_new_entrant(prev_count)

        # Release the remaining held slots and join every worker.
        for ev in releases:
            ev.set()
        for t in threads:
            t.join(timeout=10.0)
        assert stream_worker.active_stream_count() == 0
    finally:
        for ev in releases:
            ev.set()
        for t in threads:
            t.join(timeout=10.0)
        stop_sampling.set()
        sampler.join(timeout=2.0)

    # Assert 1: never more than the bounded number of concurrently-active slots.
    assert max_active[0] <= 2
    assert entered_count() == 5


def test_stream_chunks_third_blocks_until_slot_free(monkeypatch):
    """E2E sanity: 2 concurrent stream_chunks run while a 3rd blocks until a
    slot frees; all streams finish naturally and every slot is released.

    Level: end-to-end ``stream_chunks``. The mocks COMPLETE by themselves
    (no cancellation is triggered), which keeps this test deterministic on
    Windows Proactor - cancel_stream() on a freshly-stopped loop is covered at
    the worker/task level instead.
    """
    monkeypatch.setattr(config, "UI_STREAM_WORKERS", 2)
    probe = _SlotProbe()
    monkeypatch.setattr(RAGSystem, "astream", _finite_astream_factory(probe))

    threads: list[threading.Thread] = []
    for i in range(3):
        chunks = stream_chunks(f"q{i}", "test-model", f"slot-sess-{i}")
        t = threading.Thread(
            target=_consume_stream,
            args=(chunks,),
            daemon=True,
            name=f"e2e-consumer-{i}",
        )
        t.start()
        threads.append(t)

    try:
        # Two streams are live (2 slots); the 3rd bg thread is blocked waiting.
        _wait_until(
            lambda: probe.entered_count() == 2
            and stream_worker.waiting_stream_count() == 1,
            10.0,
            "2 slots must be held while the 3rd stream waits for one",
        )
        assert stream_worker.active_stream_count() == 2

        # The first stream finishes naturally (~0.8s), frees its slot, and the
        # blocked 3rd stream enters.
        _wait_until(
            lambda: probe.entered_count() == 3,
            5.0,
            "the 3rd stream must acquire a slot once the first completes",
        )
    finally:
        for t in threads:
            t.join(timeout=30.0)

    _wait_until(
        lambda: stream_worker.active_stream_count() == 0,
        5.0,
        "all slots are released after the streams finish naturally",
    )
    assert probe.entered_count() == 3
    assert len(probe.closed) == 3  # all three mocks ran their finally
    assert stream_worker.waiting_stream_count() == 0


def test_cancel_releases_semaphore():
    """Contract: cancel_stream cancels the task; the consumer-side finally then
    frees the slot, so a subsequent acquire succeeds immediately.

    Level: ``stream_worker`` contract (acquire/register/cancel/release) with a
    real task on a real event loop. The loop is driven with ``run_forever`` so
    it keeps running after the task is cancelled (mirrors a loop that hosts a
    busy + later-cancelled stream); this also keeps ``cancel_stream`` on its
    ``loop.is_running()`` path, avoiding the Windows Proactor double-run race.
    """
    assert stream_worker.acquire_stream_slot(timeout=1.0) is True
    assert stream_worker.active_stream_count() == 1

    run_id = f"cancel-release-{uuid.uuid4().hex[:8]}"
    loop = asyncio.new_event_loop()
    in_body = threading.Event()
    finished = threading.Event()

    async def hang() -> None:
        in_body.set()
        try:
            await asyncio.sleep(600)
        finally:
            finished.set()
            # Consumer-side cleanup, mirroring stream_chunks' finally: once the
            # cancelled task has drained, free the slot it was holding.
            stream_worker.unregister_stream(run_id)
            stream_worker.release_stream_slot()

    def run_forever_bg() -> None:
        asyncio.set_event_loop(loop)
        task = loop.create_task(hang())
        stream_worker.register_stream(run_id, loop, task)
        loop.run_forever()  # returns only when the test stops the loop

    bg = threading.Thread(target=run_forever_bg, daemon=True)
    try:
        bg.start()
        assert in_body.wait(5.0)
        stream_worker.cancel_stream(run_id)
        assert finished.wait(5.0), "the registered task's finally must run"
        # Slot is released by the cancelled task's consumer-side finally.
        _wait_until(
            lambda: stream_worker.active_stream_count() == 0,
            5.0,
            "cancel_stream must let the loop drain and the slot be released",
        )
        # A second stream starts immediately: the semaphore is free again.
        assert stream_worker.acquire_stream_slot(timeout=0.5) is True
        stream_worker.release_stream_slot()
    finally:
        loop.call_soon_threadsafe(loop.stop)
        bg.join(timeout=5.0)
        loop.close()


def test_ui_stream_workers_env_override(monkeypatch):
    """Provenance (Step 2.1): the UI_STREAM_WORKERS env hook via config reload.

    Level: ``common.config`` -- stream_worker reads the limit through the config
    module object, so a reload with the env var set picks up the new bound.
    """
    original = config.UI_STREAM_WORKERS
    try:
        monkeypatch.setenv("UI_STREAM_WORKERS", "2")
        importlib.reload(config)
        assert config.UI_STREAM_WORKERS == 2
    finally:
        # The reload bakes the env-derived value into the module constant;
        # restore the pre-test value so later tests read the original bound.
        config.UI_STREAM_WORKERS = original
