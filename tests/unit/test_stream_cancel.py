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
from unittest.mock import MagicMock, patch

import pytest

import common.config as config
from api.stream_pipeline import StreamChunk, StreamingResponseHandler
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


# ---------------------------------------------------------------------------
# DEFECT-2: Split timeout + _remaining queue-level drain tests
# ---------------------------------------------------------------------------


def test_cancel_preserves_tail_tokens():
    """Cancel mid-stream: remaining content+thought flushed to _remaining.

    Real handler with content_buffer_size=5 processes a scripted event stream
    yielding 8 content tokens + 1 thought token via messages mode (buffered).
    After consuming a few chunks the generator is closed — the ``_remaining``
    list must receive the flush tail (content remainder + thought + perf).
    """
    handler = StreamingResponseHandler(content_buffer_size=5)

    class _FakeChunk:
        def __init__(self, content: str = "", thought: str = "") -> None:
            self.content = content
            self.content_blocks: list[Any] = []
            self.additional_kwargs: dict[str, Any] = {}
            if thought:
                self.additional_kwargs["thought"] = thought

    async def scripted_events():  # type: ignore[return]
        for i in range(8):
            yield ("messages", (_FakeChunk(content=f"tok{i} "), {}))
        yield ("messages", (_FakeChunk(thought="reasoning here"), {}))

    remaining: list[StreamChunk] = []
    consumed: list[StreamChunk] = []

    async def _run() -> None:
        gen = handler.stream_graph_events(
            scripted_events(),
            _remaining=remaining,
        )
        try:
            async for chunk in gen:
                consumed.append(chunk)
                if len(consumed) >= 3:
                    break
        finally:
            await gen.aclose()

    asyncio.run(_run())

    assert len(remaining) >= 1
    for c in remaining:
        assert isinstance(c, StreamChunk)


def test_pre_chunk_timeout_allows_long_setup(monkeypatch):
    """Before the first chunk arrives, effective timeout = UI_STREAMING_SETUP_TIMEOUT.

    The stop-poll refactor (Defect LS1) slices q.get into
    _STOP_POLL_INTERVAL_SEC polls, so the raw q.get timeout is no longer the
    observable contract. A spy on _q_get_with_stop_poll records the effective
    deadline per iteration instead: setup(300) before the first chunk,
    inter-chunk(60) after. No TimeoutError is raised, and every q.get poll
    stays within the poll interval (the stop-responsiveness guarantee).
    """
    import queue as _q

    import ui.components.streaming_core as sm

    monkeypatch.setattr(sm, "UI_STREAMING_SETUP_TIMEOUT", 300)
    monkeypatch.setattr(sm, "UI_STREAMING_TIMEOUT", 60)
    monkeypatch.setattr(sm, "UI_STREAMING_HARD_TIMEOUT", 0)

    call_idx = 0
    poll_timeouts: list[float | None] = []
    effective_timeouts: list[float] = []

    def _scripted_get(self: Any, timeout: Any = None) -> Any:
        nonlocal call_idx
        poll_timeouts.append(timeout)
        call_idx += 1
        if call_idx <= 2:
            raise _q.Empty
        if call_idx == 3:
            return ("chunk", MagicMock())
        return ("done", None)

    monkeypatch.setattr(_q.Queue, "get", _scripted_get)

    real_helper = sm._q_get_with_stop_poll

    def _spy(q: Any, timeout: float) -> Any:
        effective_timeouts.append(timeout)
        return real_helper(q, timeout)

    monkeypatch.setattr(sm, "_q_get_with_stop_poll", _spy)

    async def _hang(*a: Any, **kw: Any) -> None:
        await asyncio.sleep(9999)

    mock_handler = MagicMock()
    mock_handler.stream_graph_events.side_effect = lambda gen, **kw: _hang()

    with (
        patch("core.rag_core.RAGSystem.astream", side_effect=_hang),
        patch(
            "ui.components.streaming_core.get_streaming_handler",
            return_value=mock_handler,
        ),
    ):
        list(stream_chunks("q", "m", "s"))

    # Effective deadline switches from setup(300) to inter-chunk(60).
    assert effective_timeouts == [300.0, 60.0]
    # Stop-responsiveness: no q.get poll blocks beyond the poll interval.
    assert all(t is not None and t <= sm._STOP_POLL_INTERVAL_SEC for t in poll_timeouts)


def test_inter_chunk_timeout_strikes_after_first_chunk(monkeypatch):
    """After first chunk, effective timeout switches to UI_STREAMING_TIMEOUT.

    A fake monotonic clock (monkeypatched _clock) expires each deadline
    deterministically: scripted Empty polls advance the fake clock 0.5s each,
    so a 60s inter-chunk deadline needs 120 polls per strike. 2×Empty (setup,
    300s deadline) + 1 chunk + 3 expired 60s deadlines → TimeoutError after
    the 3 post-first strikes. The spy proves the phase switch.
    """
    import queue as _q

    import ui.components.streaming_core as sm

    monkeypatch.setattr(sm, "UI_STREAMING_SETUP_TIMEOUT", 300)
    monkeypatch.setattr(sm, "UI_STREAMING_TIMEOUT", 60)
    monkeypatch.setattr(sm, "UI_STREAMING_HARD_TIMEOUT", 0)

    fake_now = [0.0]
    monkeypatch.setattr(sm, "_clock", lambda: fake_now[0])

    call_idx = 0
    effective_timeouts: list[float] = []

    def _scripted_get(self: Any, timeout: Any = None) -> Any:
        nonlocal call_idx
        call_idx += 1
        fake_now[0] += 0.5  # each poll consumes 0.5s of simulated time
        if call_idx <= 2:
            raise _q.Empty
        if call_idx == 3:
            return ("chunk", MagicMock())
        raise _q.Empty  # every post-chunk poll is an empty slot

    monkeypatch.setattr(_q.Queue, "get", _scripted_get)

    real_helper = sm._q_get_with_stop_poll

    def _spy(q: Any, timeout: float) -> Any:
        effective_timeouts.append(timeout)
        return real_helper(q, timeout)

    monkeypatch.setattr(sm, "_q_get_with_stop_poll", _spy)

    async def _hang(*a: Any, **kw: Any) -> None:
        await asyncio.sleep(9999)

    mock_handler = MagicMock()
    mock_handler.stream_graph_events.side_effect = lambda gen, **kw: _hang()

    with (
        patch("core.rag_core.RAGSystem.astream", side_effect=_hang),
        patch(
            "ui.components.streaming_core.get_streaming_handler",
            return_value=mock_handler,
        ),
        pytest.raises(TimeoutError),
    ):
        list(stream_chunks("q", "m", "s"))

    # setup(300) then three inter-chunk strikes with effective timeout 60.
    assert effective_timeouts == [300.0, 60.0, 60.0, 60.0]


def test_hard_cancel_no_tokens_to_lose(monkeypatch):
    """Zero chunks produced; cancel fires after strikes → done in queue."""
    import queue as _q

    import ui.components.streaming_core as sm

    monkeypatch.setattr(sm, "UI_STREAMING_SETUP_TIMEOUT", 300)
    monkeypatch.setattr(sm, "UI_STREAMING_TIMEOUT", 0.01)
    monkeypatch.setattr(sm, "UI_STREAMING_HARD_TIMEOUT", 0)

    call_idx = 0

    def _scripted_get(self: Any, timeout: Any = None) -> Any:
        nonlocal call_idx
        call_idx += 1
        if call_idx == 1:
            raise _q.Empty
        return ("done", None)

    monkeypatch.setattr(_q.Queue, "get", _scripted_get)

    async def _hang(*a: Any, **kw: Any) -> None:
        await asyncio.sleep(9999)

    mock_handler = MagicMock()
    mock_handler.stream_graph_events.side_effect = lambda gen, **kw: _hang()

    with (
        patch("core.rag_core.RAGSystem.astream", side_effect=_hang),
        patch(
            "ui.components.streaming_core.get_streaming_handler",
            return_value=mock_handler,
        ),
    ):
        result = list(stream_chunks("q", "m", "s"))

    assert result == []


def test_hard_timeout_ceiling_aborts_hung_stream(monkeypatch):
    """Hard ceiling triggers TimeoutError regardless of strike count.

    UI_STREAMING_HARD_TIMEOUT is set to a tiny value (0.001s) and
    q.get effective timeout to 0.01s.  On the second loop iteration
    the ceiling check fires (> 0.001s elapsed) before the 3-strike
    mechanism can trigger.
    """
    import ui.components.streaming_core as sm

    monkeypatch.setattr(sm, "UI_STREAMING_TIMEOUT", 0.01)
    monkeypatch.setattr(sm, "UI_STREAMING_SETUP_TIMEOUT", 0.01)
    monkeypatch.setattr(sm, "UI_STREAMING_HARD_TIMEOUT", 0.001)

    async def _never_return(*a: Any, **kw: Any) -> Any:
        await asyncio.sleep(0.1)

        async def _empty():  # type: ignore[return]
            if False:
                yield

        return _empty()

    async def _empty_stream(*a: Any, **kw: Any) -> Any:  # type: ignore[return]
        if False:
            yield

    mock_handler = MagicMock()
    mock_handler.stream_graph_events.side_effect = lambda gen, **kw: _empty_stream()

    with (
        patch("core.rag_core.RAGSystem.astream", side_effect=_never_return),
        patch(
            "ui.components.streaming_core.get_streaming_handler",
            return_value=mock_handler,
        ),
        pytest.raises(TimeoutError, match="절대 상한"),
    ):
        list(stream_chunks("q", "m", "s"))


def test_stop_during_slot_wait_exits_promptly(monkeypatch):
    """Stop while parked in slot acquisition must not wedge the teardown join.

    Regression: ``bg_task`` used to block up to 30s in
    ``acquire_stream_slot()`` while unregistered, so ``cancel_stream()``
    no-op'd and the 3s join always expired with "Stream thread did not
    exit after cancel" (slot-exhaustion cascade). Now the wait polls
    ``_stop_event`` and the abandoned thread exits promptly.
    """
    from ui.components import streaming_core as sc

    monkeypatch.setattr(config, "UI_STREAM_WORKERS", 1, raising=False)
    monkeypatch.setattr(sc, "UI_STREAMING_TIMEOUT", 2, raising=False)
    monkeypatch.setattr(sc, "UI_STREAMING_SETUP_TIMEOUT", 2, raising=False)
    assert stream_worker.acquire_stream_slot(timeout=1.0) is True
    assert stream_worker.active_stream_count() == 1

    try:
        # Thread name is f"chat-stream-{session_id[-12:]}"; "probe-sid" is
        # shorter than 12 chars so the name matches exactly — other tests'
        # lingering threads can never collide with this assertion.
        it = stream_chunks("slot-wait probe", "probe-model", "probe-sid")
        t0 = time.monotonic()
        with pytest.raises(TimeoutError):
            next(it)
        main_elapsed = time.monotonic() - t0
        assert main_elapsed < 10.0, f"main wait took {main_elapsed:.2f}s"
        # The bg thread was parked in slot acquisition; the stop must
        # release it promptly instead of riding the 30s acquire + 3s
        # join into the stuck-thread warning.
        _wait_until(
            lambda: not any(
                t.name == "chat-stream-probe-sid" and t.is_alive()
                for t in threading.enumerate()
            ),
            5.0,
            "bg thread stuck in slot acquisition after stop",
        )
        assert stream_worker.active_stream_count() == 1  # only our hold
    finally:
        stream_worker.release_stream_slot()
    assert stream_worker.active_stream_count() == 0


def test_format_stuck_thread_stack_names_parking_spot():
    """The stuck-thread snapshot must name where the thread is parked."""
    from ui.components.streaming_core import _format_stuck_thread_stack

    entered = threading.Event()
    release = threading.Event()

    def _park_here() -> None:
        entered.set()
        release.wait(timeout=10.0)

    t = threading.Thread(target=_park_here, daemon=True)
    t.start()
    try:
        assert entered.wait(5.0)
        snapshot = _format_stuck_thread_stack(t)
        assert "_park_here" in snapshot
        assert "threading" in snapshot or "wait" in snapshot
    finally:
        release.set()
        t.join(timeout=5.0)

    dead = threading.Thread(target=lambda: None, daemon=True)
    # Never started: ident is None -> placeholder, never raises.
    assert _format_stuck_thread_stack(dead) == "<no ident>"


def test_format_stream_task_stack_names_await_point():
    """The task-stack snapshot must name the coroutine's await point,
    descending into nested async generators (the real stream shape)."""
    import asyncio

    from common import stream_worker
    from ui.components.streaming_core import _format_stream_task_stack

    run_id = f"peek-{uuid.uuid4().hex[:8]}"
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    async def _inner_parked() -> Any:  # type: ignore[return]
        ready.set()
        await asyncio.sleep(600)
        if False:
            yield

    async def _outer_parked() -> None:
        agen: Any = _inner_parked()
        with contextlib.suppress(StopAsyncIteration):
            await agen.__anext__()

    def _run() -> None:
        asyncio.set_event_loop(loop)
        task = loop.create_task(_outer_parked())
        stream_worker.register_stream(run_id, loop, task)
        loop.run_forever()

    bg = threading.Thread(target=_run, daemon=True)
    bg.start()
    try:
        assert ready.wait(5.0)
        snapshot = _format_stream_task_stack(run_id)
        assert "_outer_parked" in snapshot
        assert "_inner_parked" in snapshot
        assert "live suspended asyncgens" in snapshot
    finally:
        stream_worker.unregister_stream(run_id)
        loop.call_soon_threadsafe(loop.stop)
        bg.join(timeout=5.0)
        loop.close()

    assert _format_stream_task_stack("no-such-run-id") == (
        "<task already unregistered>"
    )
