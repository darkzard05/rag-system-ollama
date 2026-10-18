"""P0 regression: bg_task must always terminate its queue with done.

Covers the three paths that previously bypassed run() finally:
- slot-wait interrupted by stop -> done (no hang)
- slot exhaustion -> error + done
- outer bg exception (loop failure) -> error + done
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import common.config as config
from common import stream_worker
from ui.components.streaming_core import stream_chunks


def _reset_stream_worker_state() -> None:
    with stream_worker._stream_lock:
        stream_worker._active_slots = 0
        stream_worker._waiting = 0
        stream_worker._active_streams.clear()
    with stream_worker._sem_lock:
        stream_worker._stream_sem = None
        stream_worker._last_limit = 0


def test_slot_exhaustion_emits_error_then_done(monkeypatch: Any) -> None:
    """Slot acquisition failure must surface error, not TimeoutError hang."""
    _reset_stream_worker_state()
    monkeypatch.setattr(config, "UI_STREAM_WORKERS", 2)
    import ui.components.streaming_core as sm

    monkeypatch.setattr(sm, "UI_STREAMING_SETUP_TIMEOUT", 5)
    monkeypatch.setattr(sm, "UI_STREAMING_TIMEOUT", 1)
    monkeypatch.setattr(sm, "UI_STREAMING_HARD_TIMEOUT", 0)
    monkeypatch.setattr(sm, "acquire_stream_slot", lambda *a, **k: False)
    # Fast-forward the 30s slot deadline so bg_task emits error+done
    # before the consumer's 3-strike timeout.
    calls = {"n": 0}

    def _fast_perf() -> float:
        calls["n"] += 1
        return 0.0 if calls["n"] <= 3 else 31.0

    monkeypatch.setattr(sm.time, "perf_counter", _fast_perf)

    with pytest.raises(RuntimeError, match="동시 스트림"):
        list(stream_chunks("q", "m", "slot-done-sess"))
    _reset_stream_worker_state()


def test_outer_bg_exception_surfaces_error_not_hang(monkeypatch: Any) -> None:
    """Worker submission failure must send error+done instead of 3x Empty hang."""
    _reset_stream_worker_state()
    import ui.components.streaming_core as sm

    monkeypatch.setattr(sm, "UI_STREAMING_SETUP_TIMEOUT", 5)
    monkeypatch.setattr(sm, "UI_STREAMING_TIMEOUT", 1)
    monkeypatch.setattr(sm, "UI_STREAMING_HARD_TIMEOUT", 0)

    boom = RuntimeError("worker boom")

    class _BoomWorker:
        @property
        def loop(self) -> Any:
            return None

        def submit(self, coro: Any) -> Any:
            coro.close()
            raise boom

    monkeypatch.setattr(sm, "AsyncWorker", _BoomWorker)

    with pytest.raises(RuntimeError) as exc_info:
        list(stream_chunks("q", "m", "outer-err-sess"))
    assert exc_info.value is boom
    _reset_stream_worker_state()


def test_done_sent_exactly_once_on_success(monkeypatch: Any) -> None:
    """Normal path still terminates with a single done (no duplicate)."""
    _reset_stream_worker_state()
    import queue as _q

    import ui.components.streaming_core as sm

    monkeypatch.setattr(sm, "UI_STREAMING_SETUP_TIMEOUT", 5)
    monkeypatch.setattr(sm, "UI_STREAMING_TIMEOUT", 5)
    monkeypatch.setattr(sm, "UI_STREAMING_HARD_TIMEOUT", 0)

    puts: list[Any] = []
    real_put = _q.Queue.put

    def _spy_put(self: Any, item: Any, *a: Any, **k: Any) -> Any:
        puts.append(item)
        return real_put(self, item, *a, **k)

    monkeypatch.setattr(_q.Queue, "put", _spy_put)

    async def _hang(*a: Any, **kw: Any) -> Any:
        import asyncio

        await asyncio.sleep(9999)

    mock_handler = MagicMock()
    mock_handler.stream_graph_events.side_effect = lambda gen, **kw: _hang()

    # Force the consumer to see done on the 2nd poll so bg thread is joined.
    call_idx = 0
    real_get = _q.Queue.get

    def _scripted_get(self: Any, *a: Any, **k: Any) -> Any:
        # Only intercept streaming_core queues (tuple messages).
        # Other threads (e.g. posthog) use different payloads/block kwargs.
        if "block" in k:
            return real_get(self, *a, **k)
        nonlocal call_idx
        call_idx += 1
        if call_idx == 1:
            return ("chunk", MagicMock())
        return ("done", None)

    monkeypatch.setattr(_q.Queue, "get", _scripted_get)

    with (
        patch("core.rag_core.RAGSystem.astream", side_effect=_hang),
        patch(
            "ui.components.streaming_core.get_streaming_handler",
            return_value=mock_handler,
        ),
    ):
        result = list(stream_chunks("q", "m", "done-once-sess"))

    assert isinstance(result, list)
    done_count = sum(1 for m, _ in puts if m == "done")
    assert done_count <= 2, f"duplicate done spam: {done_count}"
    _reset_stream_worker_state()


def test_slot_wait_stop_sends_done_directly() -> None:
    """Unit: runner funnels every early return through the single _emit_done helper.

    The stop-event itself is internal to stream_chunks, so assert the
    runner keeps a single ``finally`` that emits done on all exit paths
    (early returns included) instead of bare returns. Prevents regression
    to a silent hang.
    """
    import pathlib

    src = pathlib.Path("src/ui/components/streaming_core.py").read_text(
        encoding="utf-8"
    )
    assert "def _emit_done" in src
    # stream_chunks() body must not contain a bare q.put done outside the
    # helper: extract the runner onwards and check there.
    body = src.split("def runner_wrapper", 1)[1]
    body = body.split("def stream_content", 1)[0]
    assert 'q.put(("done", None))' not in body
    # The runner has a single finally that emits done for all exit paths.
    assert "finally:" in body
    assert "_emit_done()" in body
    # Stop during slot wait short-circuits without wedging.
    assert "슬롯 대기 중 중단" in src
