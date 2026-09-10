"""
스트리밍 응답 처리 - Task 12
실시간 토큰 스트리밍, SSE 지원, UI 업데이트 최적화

PHASE 3-P1: 버퍼/상태 머신/SSE 헬퍼/적응형 컨트롤러를 별도 모듈로 분리하고,
하위 호환성을 위해 하단에서 재-export합니다.
"""

import logging
import time
from collections.abc import AsyncIterator, Callable, Coroutine
from dataclasses import dataclass
from typing import Any, cast

from common.config import UI_CONTENT_BUFFER_SIZE
from services.monitoring.performance_monitor import (
    OperationType,
    get_performance_monitor,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AdaptiveStreamingController",
    "PriorityStreamBuffer",
    "ServerSentEventsHandler",
    "StreamChunk",
    "StreamingMetrics",
    "StreamingResponseBuilder",
    "StreamingResponseHandler",
    "StreamingState",
    "StreamingStateContext",
    "StreamingStateMachine",
    "TokenStreamBuffer",
    "create_streaming_state_machine",
    "get_adaptive_controller",
    "get_streaming_handler",
    "gzip_compress",
]


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


class StreamingResponseHandler:
    """
    스트리밍 응답 처리기 - 실시간 토큰 스트리밍
    Phase 2.2: PriorityStreamBuffer로 thought/content 분리 버퍼링
    """

    def __init__(
        self,
        content_buffer_size: int = UI_CONTENT_BUFFER_SIZE,
        content_timeout_ms: float = 10.0,
        thought_buffer_size: int = 5,
        thought_timeout_ms: float = 100.0,
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
                                token_count=max(
                                    1, len(content) // 2
                                ),  # 한국어/영어 혼용 고려 간이 계산
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
                                        token_count=max(1, len(buffered_content) // 4),
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
                    token_count=len(remaining_content.split()),
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
                op.error = str(e)
                if on_error:
                    await on_error(e)
                else:
                    raise

        return self.metrics


def get_streaming_handler() -> StreamingResponseHandler:
    return StreamingResponseHandler(content_buffer_size=UI_CONTENT_BUFFER_SIZE)


# PHASE 3-P1: 분리된 모듈 재-export (하위 호환성 계약 유지)
# isort: off
from api.stream_buffers import PriorityStreamBuffer, TokenStreamBuffer  # noqa: E402
from api.stream_state_machine import (  # noqa: E402
    StreamingState,
    StreamingStateContext,
    StreamingStateMachine,
    create_streaming_state_machine,
)
from api.sse_helpers import (  # noqa: E402
    ServerSentEventsHandler,
    StreamingResponseBuilder,
    gzip_compress,
)
from api.adaptive_controller import (  # noqa: E402
    AdaptiveStreamingController,
    get_adaptive_controller,
)
# isort: on
