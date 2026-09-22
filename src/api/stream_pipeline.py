"""API 스트리밍 파이프라인 — 단일 진실 소스 (Single Source of Truth).

Phase 2C: ``stream_buffers`` / ``stream_events`` / ``stream_state_machine`` /
``streaming_handler`` 4개 모듈을 하나로 통합한 파이프라인. ``sse_helpers`` 는
SSE 직렬화 전담으로 유지된다 (5→2).

- SSE 이벤트명 / error-fallback / cancel·dedup 동작은 byte-identical 유지.
- 라우트(path/method/status)·auth 변경 없음.
- 구경로(``stream_buffers`` / ``stream_events`` / ``stream_state_machine`` /
  ``streaming_handler``)는 순수 re-export shim 으로 유지된다.
- 취소/슬롯 게이팅은 ``common.stream_worker`` 파사드(Task1)를 통해서만
  수행한다 (본 모듈이 API 측 진입점).
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, cast

from api.adaptive_controller import (
    AdaptiveStreamingController,
    get_adaptive_controller,
)
from api.sse_helpers import (
    ServerSentEventsHandler,
    StreamingResponseBuilder,
    gzip_compress,
)
from common.config import (
    UI_CONTENT_BUFFER_SIZE,
    UI_STREAMING_CONTENT_TIMEOUT_MS,
    UI_STREAMING_THOUGHT_BUFFER_SIZE,
    UI_STREAMING_THOUGHT_TIMEOUT_MS,
)
from common.stream_worker import (
    acquire_stream_slot,
    active_stream_count,
    cancel_stream,
    register_stream,
    release_stream_slot,
    unregister_stream,
    waiting_stream_count,
)
from services.monitoring.performance_monitor import (
    OperationType,
    get_performance_monitor,
)

logger = logging.getLogger(__name__)

# Contract: streaming_handler shim 에서 재수출되던 15개 이름 + 이벤트 계약 +
# 워커 파사드(TASK1) 진입점을 모두 본 모듈에서 제공한다.
__all__ = ["AdaptiveStreamingController", "PriorityStreamBuffer", "STREAM_EVENT_TYPES", "ServerSentEventsHandler", "StreamChunk", "StreamEvent", "StreamingMetrics", "StreamingResponseBuilder", "StreamingResponseHandler", "StreamingState", "StreamingStateContext", "StreamingStateMachine", "TokenStreamBuffer", "acquire_stream_slot", "active_stream_count", "cancel_stream", "cancel_stream_run", "chunk_to_stream_events", "create_streaming_state_machine", "get_adaptive_controller", "get_streaming_handler", "gzip_compress", "register_stream", "release_stream_slot", "unregister_stream", "waiting_stream_count"]  # fmt: skip


# ---------------------------------------------------------------------------
# Chunk / metrics (구 streaming_handler.py — verbatim)
# ---------------------------------------------------------------------------


@dataclass
class StreamChunk:
    """스트리밍 청크 정보"""

    content: str = ""
    timestamp: float = 0.0
    token_count: int = 0
    chunk_index: int = 0
    is_final: bool = False
    is_status_update: bool = False  # 상태 업데이트 여부 명시
    status: str | None = None  # 현재 상태 메시지
    node_name: str | None = None  # 노드 이름 추가
    thought: str = ""  # 사고 과정 필드 기본값으로 빈 문자열
    is_thought: bool = False  # 사고 과정 청크 여부 (우선순위 버퍼용)
    metadata: dict[str, Any] | None = None  # 메타데이터 추가
    performance: dict[str, Any] | None = None  # 통합 성능 통계 추가
    raw_json: bool = False  # 구조화 모드 원시 JSON 스트리밍 플래그
    citations: list[dict[str, Any]] | None = None  # P3: 구조화 인용 배열


@dataclass
class StreamingMetrics:
    """스트리밍 성능 메트릭"""

    total_tokens: int = 0
    total_time: float = 0.0
    tokens_per_second: float = 0.0
    chunk_count: int = 0
    first_token_latency: float = 0.0
    avg_chunk_size: float = 0.0
    min_latency: float = float("inf")
    max_latency: float = 0.0


def _estimate_tokens(text: str) -> int:
    """토큰 수 추정 — 한국어/영어 혼용 고려 간이 계산 (char//2)."""
    return max(1, len(text) // 2)


# ---------------------------------------------------------------------------
# Buffers (구 stream_buffers.py — verbatim)
# ---------------------------------------------------------------------------


class TokenStreamBuffer:
    """
    토큰 버퍼 - 효율적인 버퍼링 및 배치 처리
    """

    def __init__(self, buffer_size: int = 10, timeout_ms: float = 100.0):
        self.buffer_size = buffer_size
        self.timeout_ms = timeout_ms
        self.buffer: list[str] = []
        self.last_flush_time: float = time.time()
        self.token_count = 0  # [추가] 처리된 누적 토큰 수 추적
        self.is_first_token: bool = True

    def add_token(self, token: str) -> str | None:
        self.buffer.append(token)
        self.token_count += 1
        current_time = time.time()

        # [최적화] 첫 토큰은 버퍼링 없이 즉시 전송 (TTFT 우선)
        if self.is_first_token:
            self.is_first_token = False
            return self.flush()

        if (len(self.buffer) >= self.buffer_size) or (
            (current_time - self.last_flush_time) * 1000 >= self.timeout_ms
        ):
            return self.flush()

        return None

    def flush(self) -> str | None:
        if not self.buffer:
            return None

        content = "".join(self.buffer)
        self.buffer.clear()
        self.last_flush_time = time.time()
        return content

    def reset(self) -> None:
        self.buffer.clear()
        self.last_flush_time = time.time()
        self.is_first_token = True
        self.token_count = 0


class PriorityStreamBuffer:
    """
    우선순위 기반 이중 버퍼 - Content(고우선) vs Thought(저우선) 분리
    Phase 2.2: thought가 content 블로킹하지 않도록 분리
    """

    def __init__(
        self,
        content_buffer_size: int = 1,
        content_timeout_ms: float = 10.0,
        thought_buffer_size: int = 5,
        thought_timeout_ms: float = 100.0,
    ):
        # Content 버퍼: 즉시 플러시 지향 — 크기는 호출자가 config(UI_CONTENT_BUFFER_SIZE)로 지정 (기본 1 = 토큰마다 즉시).
        self.content_buffer = TokenStreamBuffer(content_buffer_size, content_timeout_ms)
        # Thought 버퍼: 배치 처리 (size=5, timeout=100ms)
        self.thought_buffer = TokenStreamBuffer(thought_buffer_size, thought_timeout_ms)

    # 호환성 속성 (adaptive controller용)
    @property
    def buffer_size(self) -> int:
        return self.content_buffer.buffer_size

    @buffer_size.setter
    def buffer_size(self, value: int) -> None:
        self.content_buffer.buffer_size = value

    def add_token(self, token: str) -> str | None:
        """호환용: content 버퍼에 추가"""
        return self.content_buffer.add_token(token)

    def add_content(self, token: str) -> str | None:
        """Content 토큰 추가 - 즉시 반환"""
        return self.content_buffer.add_token(token)

    def add_thought(self, token: str) -> str | None:
        """Thought 토큰 추가 - 배치 반환"""
        return self.thought_buffer.add_token(token)

    def flush(self) -> str | None:
        """호환용: content 버퍼 플러시"""
        return self.content_buffer.flush()

    def flush_content(self) -> str | None:
        return self.content_buffer.flush()

    def flush_thought(self) -> str | None:
        return self.thought_buffer.flush()

    def flush_all(self) -> tuple[str | None, str | None]:
        """두 버퍼 모두 플러시"""
        return self.flush_content(), self.flush_thought()

    def reset(self) -> None:
        self.content_buffer.reset()
        self.thought_buffer.reset()


# ---------------------------------------------------------------------------
# State machine (구 stream_state_machine.py — verbatim)
# ---------------------------------------------------------------------------


class StreamingState(Enum):
    """스트리밍 상태 머신의 상태"""

    IDLE = "idle"
    INITIALIZING = "initializing"
    STREAMING = "streaming"
    BUFFERING = "buffering"
    FLUSHING = "flushing"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class StreamingStateContext:
    """스트리밍 상태 머신의 컨텍스트"""

    state: StreamingState = StreamingState.IDLE
    current_chunk: StreamChunk | None = None
    buffer: PriorityStreamBuffer | None = None
    error: Exception | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # 전이 이력 (디버그용)
    transition_history: list[tuple[StreamingState, StreamingState, float]] = field(
        default_factory=list
    )

    def transition_to(self, new_state: StreamingState) -> None:
        """상태 전이 수행"""
        old_state = self.state
        self.state = new_state
        self.transition_history.append((old_state, new_state, time.time()))
        logger.debug(
            f"[StreamingStateMachine] 상태 전이: {old_state.value} -> {new_state.value}"
        )


class StreamingStateMachine:
    """
    스트리밍 상태 머신 - Phase 2.5
    명시적 상태 전이로 스트리밍 라이프사이클 관리
    """

    # 유효한 상태 전이 정의
    VALID_TRANSITIONS = {
        StreamingState.IDLE: [StreamingState.INITIALIZING, StreamingState.CANCELLED],
        StreamingState.INITIALIZING: [
            StreamingState.STREAMING,
            StreamingState.ERROR,
            StreamingState.CANCELLED,
        ],
        StreamingState.STREAMING: [
            StreamingState.BUFFERING,
            StreamingState.FLUSHING,
            StreamingState.COMPLETED,
            StreamingState.ERROR,
            StreamingState.CANCELLED,
        ],
        StreamingState.BUFFERING: [
            StreamingState.STREAMING,
            StreamingState.FLUSHING,
            StreamingState.ERROR,
            StreamingState.CANCELLED,
        ],
        StreamingState.FLUSHING: [
            StreamingState.STREAMING,
            StreamingState.COMPLETED,
            StreamingState.ERROR,
            StreamingState.CANCELLED,
        ],
        StreamingState.COMPLETED: [StreamingState.IDLE],
        StreamingState.ERROR: [StreamingState.IDLE],
        StreamingState.CANCELLED: [StreamingState.IDLE],
    }

    def __init__(self, context: StreamingStateContext | None = None):
        self.context = context or StreamingStateContext()

    def can_transition(self, new_state: StreamingState) -> bool:
        """전이 가능 여부 확인"""
        return new_state in self.VALID_TRANSITIONS.get(self.context.state, [])

    def transition(self, new_state: StreamingState) -> bool:
        """상태 전이 시도 (성공 시 True, 실패 시 False)"""
        if self.can_transition(new_state):
            self.context.transition_to(new_state)
            return True
        logger.warning(
            f"[StreamingStateMachine] 유효하지 않은 전이: {self.context.state.value} -> {new_state.value}"
        )
        return False

    def force_transition(self, new_state: StreamingState) -> None:
        """강제 상태 전이 (에러 복구 등)"""
        self.context.transition_to(new_state)

    def initialize(self, buffer: PriorityStreamBuffer) -> bool:
        """스트리밍 초기화"""
        if self.transition(StreamingState.INITIALIZING):
            self.context.buffer = buffer
            return self.transition(StreamingState.STREAMING)
        return False

    def on_chunk_received(self, chunk: StreamChunk) -> bool:
        """청크 수신 시 호출"""
        if self.context.state == StreamingState.STREAMING:
            self.context.current_chunk = chunk
            return True
        return False

    def on_buffer_full(self) -> bool:
        """버퍼 가득 참 시 호출"""
        return self.transition(StreamingState.BUFFERING)

    def on_flush_start(self) -> bool:
        """플러시 시작 시 호출"""
        return self.transition(StreamingState.FLUSHING)

    def on_flush_complete(self) -> bool:
        """플러시 완료 시 호출"""
        if self.context.state == StreamingState.FLUSHING:
            return self.transition(StreamingState.STREAMING)
        return False

    def complete(self) -> bool:
        """스트리밍 완료"""
        if self.transition(StreamingState.FLUSHING):
            return self.transition(StreamingState.COMPLETED)
        return False

    def error(self, error: Exception) -> bool:
        """에러 발생"""
        self.context.error = error
        return self.transition(StreamingState.ERROR)

    def cancel(self) -> bool:
        """스트리밍 취소"""
        return self.transition(StreamingState.CANCELLED)

    def reset(self) -> None:
        """상태 머신 리셋"""
        self.context = StreamingStateContext()

    def get_state(self) -> StreamingState:
        """현재 상태 반환"""
        return self.context.state

    def get_history(self) -> list[tuple[str, str, float]]:
        """전이 이력 반환"""
        return [
            (old.value, new.value, ts)
            for old, new, ts in self.context.transition_history
        ]


# 편의 함수
def create_streaming_state_machine() -> StreamingStateMachine:
    """스트리밍 상태 머신 생성"""
    return StreamingStateMachine()


# ---------------------------------------------------------------------------
# Response handler (구 streaming_handler.py — verbatim)
# ---------------------------------------------------------------------------


class StreamingResponseHandler:
    """
    스트리밍 응답 처리기 - 실시간 토큰 스트리밍
    Phase 2.2: PriorityStreamBuffer로 thought/content 분리 버퍼링
    """

    def __init__(
        self,
        content_buffer_size: int = UI_CONTENT_BUFFER_SIZE,
        content_timeout_ms: float = UI_STREAMING_CONTENT_TIMEOUT_MS,
        thought_buffer_size: int = UI_STREAMING_THOUGHT_BUFFER_SIZE,
        thought_timeout_ms: float = UI_STREAMING_THOUGHT_TIMEOUT_MS,
    ):
        self.buffer = PriorityStreamBuffer(
            content_buffer_size,
            content_timeout_ms,
            thought_buffer_size,
            thought_timeout_ms,
        )
        self.metrics = StreamingMetrics()
        self.chunk_index = 0
        self.start_time: float | None = None
        self.first_token_time: float | None = None
        self.last_chunk_time: float | None = None
        self.node_metadata: dict[str, Any] = {}
        self._last_chunk_raw_json: bool = False

    async def stream_graph_events(
        self,
        event_stream: AsyncIterator[tuple[str, Any]],
        adaptive_controller: Any = None,
        *,
        _remaining: list[StreamChunk] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        astream(stream_mode=["messages", "custom"])의 이벤트를 소비하여
        가공된 스트리밍 청크를 생성합니다.
        """
        from contextlib import aclosing

        self.start_time = time.time()
        self.last_chunk_time = self.start_time
        self.chunk_index = 0
        self.metrics = StreamingMetrics()
        self.first_token_time = None
        self.buffer.reset()
        self.node_metadata = {}
        self._last_chunk_raw_json = False
        # Adaptive controller와의 호환성을 위해 버퍼 크기 속성 제공
        self.buffer_size = self.buffer.content_buffer.buffer_size

        try:
            async with aclosing(cast(Any, event_stream)) as stream:
                async for mode, data in stream:
                    current_time = time.time()

                    if mode == "custom":
                        status = data.get("status")
                        if status:
                            yield StreamChunk(
                                content="",
                                timestamp=current_time,
                                token_count=0,
                                chunk_index=self.chunk_index,
                                is_status_update=True,
                                status=status,
                            )
                            self.chunk_index += 1

                        if "documents" in data:
                            yield StreamChunk(
                                content="",
                                timestamp=current_time,
                                token_count=0,
                                chunk_index=self.chunk_index,
                                metadata={"documents": data["documents"]},
                            )
                            self.chunk_index += 1

                        # [추가] custom 이벤트의 content와 thought 처리
                        content = data.get("content")
                        thought = data.get("thought")
                        raw_json = bool(data.get("raw_json", False))

                        if thought:
                            yield StreamChunk(
                                content="",
                                timestamp=current_time,
                                token_count=0,
                                chunk_index=self.chunk_index,
                                thought=thought,
                                raw_json=raw_json,
                            )
                            self.chunk_index += 1

                        if content:
                            self._last_chunk_raw_json = bool(raw_json)
                            yield StreamChunk(
                                content=content,
                                timestamp=current_time,
                                token_count=_estimate_tokens(content),
                                chunk_index=self.chunk_index,
                                raw_json=raw_json,
                            )
                            self.chunk_index += 1

                        # P3: 구조화된 인용 배열(citations[])을 청크로 전달한다.
                        if "citations" in data:
                            cits = data["citations"]
                            if isinstance(cits, list):
                                yield StreamChunk(
                                    content="",
                                    timestamp=current_time,
                                    chunk_index=self.chunk_index,
                                    citations=cits,
                                )
                                self.chunk_index += 1

                    elif mode == "messages":
                        from langchain_core.messages import AIMessageChunk

                        chunk_obj, _ = data if isinstance(data, tuple) else (data, {})

                        if isinstance(chunk_obj, AIMessageChunk) or hasattr(
                            chunk_obj, "content"
                        ):
                            content = getattr(chunk_obj, "content", "")
                            thought = ""

                            if (
                                hasattr(chunk_obj, "content_blocks")
                                and chunk_obj.content_blocks
                            ):
                                for block in chunk_obj.content_blocks:
                                    if (
                                        isinstance(block, dict)
                                        and block.get("type") == "reasoning"
                                    ):
                                        thought += block.get("reasoning", "")

                            additional_kwargs = getattr(
                                chunk_obj, "additional_kwargs", {}
                            )
                            if not thought and additional_kwargs:
                                thought = (
                                    additional_kwargs.get("reasoning_content")
                                    or additional_kwargs.get("reasoning")
                                    or additional_kwargs.get("thinking")
                                    or additional_kwargs.get("thought")
                                    or ""
                                )

                            if isinstance(content, list):
                                actual_content = ""
                                for item in content:
                                    if isinstance(item, dict):
                                        if item.get("type") == "text":
                                            actual_content += item.get("text", "")
                                        elif item.get("type") == "reasoning":
                                            thought += item.get("reasoning", "")
                                    elif isinstance(item, str):
                                        actual_content += item
                                content = actual_content
                            else:
                                content = str(content)

                            if adaptive_controller and self.last_chunk_time:
                                latency_ms = (
                                    current_time - self.last_chunk_time
                                ) * 1000
                                adaptive_controller.record_latency(latency_ms)
                                self.buffer.buffer_size = (
                                    adaptive_controller.get_buffer_size()
                                )

                            self.last_chunk_time = current_time

                            if thought:
                                yield StreamChunk(
                                    content="",
                                    timestamp=current_time,
                                    token_count=0,
                                    chunk_index=self.chunk_index,
                                    thought=thought,
                                    is_thought=True,
                                )
                                self.chunk_index += 1

                            if content:
                                if self.first_token_time is None:
                                    self.first_token_time = current_time
                                self._last_chunk_raw_json = False

                                # Phase 2.2: content는 즉시 플러시되는 content 버퍼에 추가
                                buffered_content = self.buffer.add_content(content)
                                if buffered_content:
                                    chunk = StreamChunk(
                                        content=buffered_content,
                                        timestamp=current_time,
                                        token_count=_estimate_tokens(buffered_content),
                                        chunk_index=self.chunk_index,
                                    )
                                    self.metrics.total_tokens += chunk.token_count
                                    self.metrics.chunk_count += 1
                                    yield chunk
                                    self.chunk_index += 1

                    elif mode == "updates":
                        for node_name, node_output in data.items():
                            if node_name == "retrieve":
                                docs = node_output.get("relevant_docs", [])
                                if docs:
                                    yield StreamChunk(
                                        content="",
                                        timestamp=current_time,
                                        token_count=0,
                                        chunk_index=self.chunk_index,
                                        metadata={"documents": docs},
                                        status=f"관련 문서 {len(docs)}개를 찾았습니다.",
                                    )
                                    self.chunk_index += 1
                            elif node_name == "generate":
                                perf = node_output.get("performance")
                                if perf:
                                    self.node_metadata.update(perf)
                                    # or-패턴: total_tokens==0일 때만 perf 값으로 대체
                                    self.metrics.total_tokens = (
                                        self.metrics.total_tokens
                                        or perf.get("token_count", 0)
                                    )
                                    input_tokens = perf.get("input_token_count", 0)

                                    yield StreamChunk(
                                        content="",
                                        timestamp=current_time,
                                        chunk_index=self.chunk_index,
                                        performance={
                                            **perf,
                                            "total_time": self.metrics.total_time,
                                            "ttft": self.metrics.first_token_latency,
                                            "tps": self.metrics.tokens_per_second,
                                            "input_token_count": input_tokens,
                                        },
                                    )
                                    self.chunk_index += 1

        except (RuntimeError, ValueError, TypeError) as e:
            logger.error(f"[Streaming] 스트림 처리 중 오류: {e}", exc_info=True)
            raise
        finally:
            # Phase 2.2: 두 버퍼 모두 플러시
            remaining_content, remaining_thought = self.buffer.flush_all()

            final_chunk: StreamChunk | None = None
            if remaining_content:
                final_chunk = StreamChunk(
                    content=remaining_content,
                    timestamp=time.time(),
                    token_count=_estimate_tokens(remaining_content),
                    chunk_index=self.chunk_index,
                    is_final=True,
                    raw_json=self._last_chunk_raw_json,
                )
                self.metrics.total_tokens += final_chunk.token_count
                self.metrics.chunk_count += 1

            thought_chunk: StreamChunk | None = None
            if remaining_thought:
                thought_chunk = StreamChunk(
                    content="",
                    timestamp=time.time(),
                    token_count=0,
                    chunk_index=self.chunk_index,
                    thought=remaining_thought,
                    is_thought=True,
                )
                self.chunk_index += 1

            if _remaining is not None:
                if final_chunk is not None:
                    _remaining.append(final_chunk)
                if thought_chunk is not None:
                    _remaining.append(thought_chunk)
            else:
                if final_chunk is not None:
                    yield final_chunk
                if thought_chunk is not None:
                    yield thought_chunk

            self.metrics.total_time = time.time() - (self.start_time or time.time())
            if self.first_token_time and self.start_time:
                self.metrics.first_token_latency = (
                    self.first_token_time - self.start_time
                )
            if self.metrics.total_time > 0:
                self.metrics.tokens_per_second = (
                    self.metrics.total_tokens / self.metrics.total_time
                )

            final_performance = {
                **self.node_metadata,
                "total_time": self.metrics.total_time,
                "ttft": self.metrics.first_token_latency,
                "tps": self.metrics.tokens_per_second,
                "token_count": self.metrics.total_tokens,
            }

            perf_chunk = StreamChunk(
                content="",
                timestamp=time.time(),
                is_final=True,
                performance=final_performance,
            )
            if _remaining is not None:
                _remaining.append(perf_chunk)
            else:
                yield perf_chunk

    async def stream_response(
        self,
        response_generator: AsyncIterator[str],
        on_chunk: Callable[[StreamChunk], Coroutine[Any, Any, None]],
        on_complete: Callable[[], Coroutine[Any, Any, None]] | None = None,
        on_error: Callable[[Exception], Coroutine[Any, Any, None]] | None = None,
        operation_name: str = "response_streaming",
        adaptive_controller: Any = None,
    ) -> StreamingMetrics:
        self.start_time = time.time()
        self.metrics = StreamingMetrics()
        self.chunk_index = 0

        with get_performance_monitor().track_operation(
            OperationType.LLM_INFERENCE,
            {"stage": "streaming", "buffer_size": self.buffer.buffer_size},
        ) as op:
            try:
                async for token in response_generator:
                    if adaptive_controller:
                        new_size = adaptive_controller.get_buffer_size()
                        if self.buffer.buffer_size != new_size:
                            self.buffer.buffer_size = new_size

                    if self.first_token_time is None:
                        self.first_token_time = time.time()
                        self.metrics.first_token_latency = (
                            self.first_token_time - self.start_time
                        )

                    # Phase 2.2: content 버퍼에 추가 (즉시 플러시)
                    buffered_content = self.buffer.add_content(token)

                    if buffered_content:
                        chunk = StreamChunk(
                            content=buffered_content,
                            timestamp=time.time(),
                            token_count=len(buffered_content.split()),
                            chunk_index=self.chunk_index,
                            is_final=False,
                        )

                        self.metrics.total_tokens += chunk.token_count
                        self.metrics.chunk_count += 1

                        latency = chunk.timestamp - self.start_time
                        self.metrics.min_latency = min(
                            self.metrics.min_latency, latency
                        )
                        self.metrics.max_latency = max(
                            self.metrics.max_latency, latency
                        )

                        await on_chunk(chunk)
                        self.chunk_index += 1

                # Phase 2.2: 두 버퍼 모두 플러시
                remaining_content, remaining_thought = self.buffer.flush_all()
                if remaining_content:
                    final_chunk = StreamChunk(
                        content=remaining_content,
                        timestamp=time.time(),
                        token_count=len(remaining_content.split()),
                        chunk_index=self.chunk_index,
                        is_final=True,
                    )
                    self.metrics.total_tokens += final_chunk.token_count
                    self.metrics.chunk_count += 1
                    await on_chunk(final_chunk)

                self.metrics.total_time = time.time() - self.start_time
                self.metrics.tokens_per_second = (
                    self.metrics.total_tokens / self.metrics.total_time
                    if self.metrics.total_time > 0
                    else 0
                )
                self.metrics.avg_chunk_size = (
                    self.metrics.total_tokens / self.metrics.chunk_count
                    if self.metrics.chunk_count > 0
                    else 0
                )

                if on_complete:
                    await on_complete()

                op.tokens = self.metrics.total_tokens

            except (RuntimeError, ValueError) as e:
                logger.error(f"[Streaming] 에러: {e}")
                op.error = str(e)  # type: ignore[attr-defined]
                if on_error:
                    await on_error(e)
                else:
                    raise

        return self.metrics


def get_streaming_handler() -> StreamingResponseHandler:
    """call-time 상수 읽기 seam: ``UI_CONTENT_BUFFER_SIZE`` 패치가 반영된다."""
    return StreamingResponseHandler(
        content_buffer_size=UI_CONTENT_BUFFER_SIZE,
        content_timeout_ms=UI_STREAMING_CONTENT_TIMEOUT_MS,
        thought_buffer_size=UI_STREAMING_THOUGHT_BUFFER_SIZE,
        thought_timeout_ms=UI_STREAMING_THOUGHT_TIMEOUT_MS,
    )


# ---------------------------------------------------------------------------
# Events (구 stream_events.py — verbatim)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Canonical event type registry — keep in sync with StreamEvent type field.
# Contract tests validate this tuple matches the six mapping branches below.
# ---------------------------------------------------------------------------
STREAM_EVENT_TYPES: tuple[str, ...] = (
    "status",
    "message",
    "thought",
    "sources",
    "citations",
    "metrics",
)


@dataclass(frozen=True)
class StreamEvent:
    """불변 스트리밍 이벤트 컨테이너.

    Attributes:
        type: 이벤트 종류 (STREAM_EVENT_TYPES 중 하나).
        payload: 이벤트별 데이터 딕셔너리.
    """

    type: str
    payload: dict[str, Any]


def chunk_to_stream_events(chunk: StreamChunk) -> list[StreamEvent]:
    """StreamChunk를 순서가 보장된 StreamEvent 목록으로 변환합니다.

    각 필드는 독립적으로 검사되며, 하나의 Chunk에서 여러 이벤트가
    생성될 수 있습니다. 반환 순서는 항상 1→6 (status→metrics)으로 고정됩니다.

    빈 값이나 누락된 필드는 해당 이벤트를 생성하지 않습니다.

    Args:
        chunk: 변환할 스트리밍 청크.

    Returns:
        이벤트 목록 (빈 청크의 경우 빈 리스트).
    """
    events: list[StreamEvent] = []

    # 모든 필드는 getattr로 방어적으로 읽는다: 테스트는 물론,
    # 일부 필드만 가진 duck-typed 청크(SimpleNamespace 등)도
    # AttributeError 없이 처리되어야 한다 (기존 소비자 방어 패턴과 동일).
    status = getattr(chunk, "status", None)
    node_name = getattr(chunk, "node_name", None)
    content = getattr(chunk, "content", None)
    thought = getattr(chunk, "thought", None)
    metadata = getattr(chunk, "metadata", None)
    citations = getattr(chunk, "citations", None)
    performance = getattr(chunk, "performance", None)

    # 1. status — 노드 상태 메시지
    if status:
        events.append(StreamEvent("status", {"message": status, "node": node_name}))

    # 2. message — LLM 응답 콘텐츠
    if content:
        events.append(StreamEvent("message", {"content": content}))

    # 3. thought — 모델 사고 과정
    if thought:
        events.append(StreamEvent("thought", {"content": thought}))

    # 4. sources — 원본 문서 (자르기 없이 전달)
    if metadata and "documents" in metadata:
        events.append(StreamEvent("sources", {"documents": metadata["documents"]}))

    # 5. citations — 구조화된 인용
    if citations:
        events.append(StreamEvent("citations", {"citations": citations}))

    # 6. metrics — 성능 통계
    if performance:
        events.append(StreamEvent("metrics", {"metrics": performance}))

    return events


# ---------------------------------------------------------------------------
# Worker facade (Task1) — API 측 취소/슬롯 진입점
# ---------------------------------------------------------------------------


def cancel_stream_run(run_id: str) -> None:
    """``run_id`` 로 등록된 스트림 태스크를 취소한다.

    ``common.stream_worker`` 파사드(Task1)로의 위임 — API 레이어는
    워커 세마포어/레지스트리를 직접 만지지 않는다. 미등록 id 는
    파사드가 warning 후 no-op 한다 (byte-identical).
    """
    cancel_stream(run_id)
