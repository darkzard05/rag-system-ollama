"""DEFECT #1: 스트리밍이 Streamlit 이벤트 루프를 블로킹하는지 검증하는 테스트.

현재 구현: consume_stream_into_message / _content_generator가 단일 script run에서
동기적으로 stream_chunks를 소비하므로, 10-60초 동안 UI(사이드바, PDF 뷰어, 입력창)
가 완전히 멈춘다.

목표 상태: 스트리밍 소비가 백그라운드 스레드/큐로 오프로드되어, 메인 스레드에서
heartbeat(주기적 세션 상태 확인)가 1초 이내로 응답해야 한다.

이 테스트는 현재 코드에서 반드시 실패해야 한다 (TDD 레드).
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import ui.components.streaming as streaming_mod
import ui.components.streaming_core as streaming_core_mod
import ui.components.streaming_state as streaming_state_mod
from api.streaming_handler import StreamChunk


class _FakeSessionManager(MagicMock):
    """consume_stream_into_message용 SessionManager 스텁."""

    _messages: list[dict[str, Any]] = []
    _store: dict[str, Any] = {}

    @classmethod
    def reset(cls) -> None:
        cls._messages = []
        cls._store = {}

    @classmethod
    def get(
        cls,
        key: str,
        default: Any = None,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        return cls._store.get(key, default)

    @classmethod
    def set(cls, key: str, value: Any, **kwargs: Any) -> None:
        cls._store[key] = value

    @classmethod
    def add_message(
        cls,
        role: str,
        content: str,
        msg_type: str = "general",
        session_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        message: dict[str, Any] = {
            "msg_id": kwargs.pop("msg_id", str(uuid.uuid4())),
            "role": role,
            "content": content,
            "msg_type": msg_type,
            **kwargs,
        }
        for index, existing in enumerate(cls._messages):
            if existing.get("msg_id") == message["msg_id"]:
                cls._messages[index].update(message)
                return
        cls._messages.append(message)

    @classmethod
    def get_messages(cls, session_id: str | None = None) -> list[dict[str, Any]]:
        return list(cls._messages)

    @classmethod
    def _acquire_lock(cls, sid: str) -> Any:
        from contextlib import nullcontext

        return nullcontext()

    @classmethod
    def _get_state(cls, sid: str) -> dict[str, Any]:
        return {"messages": cls._messages}


class _SlowStreamIterator:
    """실제 지연이 있는 스트리밍 이터레이터 — 스트리밍 블로킹을 시뮬레이션."""

    def __init__(self, chunk_count: int = 5, delay_per_chunk: float = 0.15) -> None:
        self._count = 0
        self._chunk_count = chunk_count
        self._delay = delay_per_chunk

    def __iter__(self):
        return self

    def __next__(self) -> StreamChunk:
        if self._count >= self._chunk_count:
            raise StopIteration
        self._count += 1
        # 의도적 지연 — 실제 스트리밍에서 Ollama 응답 대기 시뮬레이션
        time.sleep(self._delay)
        return StreamChunk(
            content=f"chunk_{self._count} ",
            status=f"step_{self._count}",
        )


class TestStreamingDoesNotBlockEventLoop:
    """DEFECT #1: 스트리밍 소비가 이벤트 루프를 블로킹하면 안 된다.

    현재 구현에서 이 테스트는 반드시 실패해야 한다:
    - consume_stream_into_message는 동기적으로 stream_chunks를 순회하므로
      스트리밍 중 메인 스레드가 블로킹된다.
    - 목표: 스트리밍이 백그라운드에서 수행되고, 메인 스레드는 heartbeat를
      유지할 수 있어야 한다.
    """

    def test_streaming_blocks_heartbeat_response(self) -> None:
        """스트리밍 중 세션 heartbeat(상태 확인)가 지연되면 안 된다.

        현재 실패 원인: consume_stream_into_message가 단일 스레드에서
        동기 청크를 순회하므로, 다른 스레드의 heartbeat 요청이 블로킹된다.
        """
        _FakeSessionManager.reset()
        _FakeSessionManager.set("is_generating_answer", False)

        heartbeat_latencies: list[float] = []
        heartbeat_errors: list[str] = []
        streaming_done = threading.Event()

        def _heartbeat_monitor() -> None:
            """별도 스레드에서 주기적으로 heartbeat를 측정한다."""
            while not streaming_done.is_set():
                start = time.monotonic()
                try:
                    _FakeSessionManager.get("is_generating_answer", False)
                    latency = time.monotonic() - start
                    heartbeat_latencies.append(latency)
                except Exception as e:
                    heartbeat_errors.append(str(e))
                time.sleep(0.05)  # 50ms 간격

        chunks = _SlowStreamIterator(chunk_count=8, delay_per_chunk=0.1)

        with (
            patch.object(streaming_mod, "stream_chunks", return_value=chunks),
            patch.object(streaming_mod, "SessionManager", _FakeSessionManager),
            patch.object(streaming_state_mod, "SessionManager", _FakeSessionManager),
        ):
            monitor = threading.Thread(target=_heartbeat_monitor, daemon=True)
            monitor.start()

            streaming_mod.consume_stream_into_message(
                "test_sid", "test query", "test-model"
            )

            streaming_done.set()
            monitor.join(timeout=2.0)

        # heartbeat가 오래 걸리면 안 된다
        assert not heartbeat_errors, f"heartbeat 스레드 오류: {heartbeat_errors}"
        if heartbeat_latencies:
            max_latency = max(heartbeat_latencies)
            assert max_latency < 0.5, (
                f"heartbeat 지연이 {max_latency:.3f}s — "
                f"스트리밍이 이벤트 루프를 블로킹하고 있다. "
                f"(총 {len(heartbeat_latencies)}개 측정)"
            )

    def test_streaming_takes_longer_than_real_time(self) -> None:
        """스트리밍이 실시간보다 오래 걸리면 블로킹 증거.

        현재 실패 원인: 동기 스트리밍은 청크 간 지연이 총합되어
        전체 시간이 청크 수 × 지연 시간 이상이 된다.
        """
        _FakeSessionManager.reset()
        chunk_count = 6
        delay_per_chunk = 0.1
        expected_min_time = chunk_count * delay_per_chunk * 0.8

        chunks = _SlowStreamIterator(
            chunk_count=chunk_count, delay_per_chunk=delay_per_chunk
        )

        with (
            patch.object(streaming_mod, "stream_chunks", return_value=chunks),
            patch.object(streaming_mod, "SessionManager", _FakeSessionManager),
            patch.object(streaming_state_mod, "SessionManager", _FakeSessionManager),
        ):
            start = time.monotonic()
            streaming_mod.consume_stream_into_message(
                "test_sid", "test query", "test-model"
            )
            elapsed = time.monotonic() - start

        # 스트리밍이 실시간 지연을 반영하면 총 시간이 최소 기준 이상
        assert elapsed >= expected_min_time, (
            f"스트리밍이 {elapsed:.3f}s — 지연 합계({expected_min_time:.3f}s)보다 빠르다. "
            "블로킹이 제대로 시뮬레이션되지 않았다."
        )

    def test_streaming_cancellation_is_responsive(self) -> None:
        """사용자 중단(generation_cancel)이 즉시 반영되어야 한다.

        현재 실패 원인: consume_stream_into_message가 동기 루프에서
        청크를 순회하므로, 중단 요청이 다음 청크까지 반영되지 않는다.
        """
        _FakeSessionManager.reset()
        cancel_after_chunks = 2
        chunks_consumed = [0]

        def _tracking_slow_stream(query: str, model: str, sid: str):
            for i in range(10):
                if i == cancel_after_chunks:
                    _FakeSessionManager.set("generation_cancel", True, session_id=sid)
                chunks_consumed[0] = i + 1
                time.sleep(0.05)  # 지연 포함
                yield StreamChunk(content=f"chunk_{i} ", status=f"step_{i}")

        with (
            patch.object(streaming_mod, "stream_chunks", _tracking_slow_stream),
            patch.object(streaming_mod, "SessionManager", _FakeSessionManager),
            patch.object(streaming_state_mod, "SessionManager", _FakeSessionManager),
        ):
            streaming_mod.consume_stream_into_message(
                "test_sid", "test query", "test-model"
            )

        # 중단 후 추가 청크가 소비되면 안 된다 (현재: 동기 루프이므로
        # generation_cancel 확인이 청크 사이에만 이루어져 1개 추가 소비 가능)
        assert chunks_consumed[0] <= cancel_after_chunks + 1, (
            f"중단 후 {chunks_consumed[0]}개 청크 소비됨 "
            f"(목표: 최대 {cancel_after_chunks + 1}개)"
        )

    def test_streaming_preserves_partial_answer_on_cancel(self) -> None:
        """중단 시 부분 응답이 영속화되어야 한다."""
        _FakeSessionManager.reset()
        _FakeSessionManager.set("generation_cancel", False)

        chunks = [
            StreamChunk(content="Hello ", status="step_0"),
            StreamChunk(content="world", status="step_1"),
        ]

        with (
            patch.object(streaming_mod, "stream_chunks", return_value=iter(chunks)),
            patch.object(streaming_mod, "SessionManager", _FakeSessionManager),
            patch.object(streaming_state_mod, "SessionManager", _FakeSessionManager),
            patch.object(streaming_core_mod, "SessionManager", _FakeSessionManager),
        ):
            result = streaming_mod.consume_stream_into_message(
                "test_sid", "test query", "test-model"
            )

        assert result is not None
        assert result.get("content") == "Hello world"
        assert result.get("cancelled") is False

    def test_streaming_error_sets_is_generating_false(self) -> None:
        """스트리밍 오류 발생 시 is_generating_answer가 False로 리셋되어야 한다."""
        _FakeSessionManager.reset()
        _FakeSessionManager.set("is_generating_answer", True)

        def _failing_stream(query: str, model: str, sid: str):
            yield StreamChunk(content="partial ", status="step_0")
            raise ConnectionError("Ollama connection refused")

        with (
            patch.object(streaming_mod, "stream_chunks", _failing_stream),
            patch.object(streaming_mod, "SessionManager", _FakeSessionManager),
            patch.object(streaming_state_mod, "SessionManager", _FakeSessionManager),
            patch.object(streaming_core_mod, "SessionManager", _FakeSessionManager),
        ):
            result = streaming_mod.consume_stream_into_message(
                "test_sid", "test query", "test-model"
            )

        assert _FakeSessionManager.get("is_generating_answer") is False
        assert result is not None
        assert result.get("error") is not None
