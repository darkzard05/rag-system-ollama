"""
스트리밍 응답 처리 - Task 12
실시간 토큰 스트리밍, SSE 지원, UI 업데이트 최적화
"""

import logging
import time
from collections.abc import AsyncIterator, Callable, Coroutine
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

from services.monitoring.performance_monitor import (
    OperationType,
    get_performance_monitor,
)

logger = logging.getLogger(__name__)
monitor = get_performance_monitor()


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
    thought: str = ""  # 사고 과정 필드 기본값 빈 문자열
    metadata: dict[str, Any] | None = None  # 메타데이터 추가
    performance: dict[str, Any] | None = None  # 통합 성능 통계 추가


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

        # [최적화] 첫 5개 토큰은 버퍼링 없이 즉시 전송 (TTFT 우선)
        if self.token_count <= 5 or self.is_first_token:
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


class StreamingResponseHandler:
    """
    스트리밍 응답 처리기 - 실시간 토큰 스트리밍
    """

    def __init__(self, buffer_size: int = 1, timeout_ms: float = 30.0):
        self.buffer = TokenStreamBuffer(buffer_size, timeout_ms)
        self.metrics = StreamingMetrics()
        self.chunk_index = 0
        self.start_time: float | None = None
        self.first_token_time: float | None = None
        self.last_chunk_time: float | None = None
        self.node_metadata: dict[str, Any] = {}

    async def stream_graph_events(
        self,
        event_stream: AsyncIterator[tuple[str, Any]],
        adaptive_controller: Any = None,
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
                                thought = additional_kwargs.get("thinking", "")

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
                                )
                                self.chunk_index += 1

                            if content:
                                if self.first_token_time is None:
                                    self.first_token_time = current_time

                                buffered_content = self.buffer.add_token(content)
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

        except Exception as e:
            logger.error(f"[Streaming] 스트림 처리 중 오류: {e}", exc_info=True)
        finally:
            remaining = self.buffer.flush()
            if remaining:
                final_chunk = StreamChunk(
                    content=remaining,
                    timestamp=time.time(),
                    token_count=len(remaining.split()),
                    chunk_index=self.chunk_index,
                    is_final=True,
                )
                self.metrics.total_tokens += final_chunk.token_count
                self.metrics.chunk_count += 1
                yield final_chunk

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

            yield StreamChunk(
                content="",
                timestamp=time.time(),
                is_final=True,
                performance=final_performance,
            )

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

        with monitor.track_operation(
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

                    buffered_content = self.buffer.add_token(token)

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

                remaining = self.buffer.flush()
                if remaining:
                    final_chunk = StreamChunk(
                        content=remaining,
                        timestamp=time.time(),
                        token_count=len(remaining.split()),
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

            except Exception as e:
                logger.error(f"[Streaming] 에러: {e}")
                op.error = str(e)
                if on_error:
                    await on_error(e)
                else:
                    raise

        return self.metrics


class ServerSentEventsHandler:
    @staticmethod
    def format_sse_event(
        event_type: str, data: dict[str, Any], event_id: int | None = None
    ) -> str:
        import orjson

        lines = []
        if event_id is not None:
            lines.append(f"id: {event_id}")
        if event_type:
            lines.append(f"event: {event_type}")
        json_data = orjson.dumps(data).decode("utf-8")
        lines.append(f"data: {json_data}")
        lines.append("")
        return "\n".join(lines) + "\n"

    @staticmethod
    def format_sse_error(error_message: str, error_code: int = 500) -> str:
        data = {
            "error": error_message,
            "code": error_code,
            "timestamp": datetime.now().isoformat(),
        }
        return ServerSentEventsHandler.format_sse_event("error", data)

    @staticmethod
    def format_sse_keepalive(message: str = "keep-alive") -> str:
        return f": {message}\n\n"


class StreamingResponseBuilder:
    def __init__(self, max_buffer_size: int = 100000):
        self.chunks: list[StreamChunk] = []
        self.max_buffer_size = max_buffer_size
        self.total_content = ""

    def add_chunk(self, chunk: StreamChunk) -> None:
        if len(self.total_content) + len(chunk.content) > self.max_buffer_size:
            while self.chunks and len(self.total_content) > self.max_buffer_size * 0.8:
                removed = self.chunks.pop(0)
                self.total_content = self.total_content[len(removed.content) :]
        self.chunks.append(chunk)
        self.total_content += chunk.content

    def get_content(self) -> str:
        return self.total_content

    def get_chunks(self) -> list[StreamChunk]:
        return self.chunks

    def reset(self) -> None:
        self.chunks.clear()
        self.total_content = ""


class AdaptiveStreamingController:
    def __init__(
        self,
        initial_buffer_size: int = 1,
        min_buffer_size: int = 1,
        max_buffer_size: int = 10,
    ):
        self.current_buffer_size = initial_buffer_size
        self.min_buffer_size = min_buffer_size
        self.max_buffer_size = max_buffer_size
        self.latency_samples: list[float] = []
        self.max_samples = 50

    def record_latency(self, latency_ms: float) -> None:
        self.latency_samples.append(latency_ms)
        if len(self.latency_samples) > self.max_samples:
            self.latency_samples.pop(0)
        self._adjust_buffer_size()

    def _adjust_buffer_size(self) -> None:
        if len(self.latency_samples) < 10:
            return
        avg_latency = sum(self.latency_samples) / len(self.latency_samples)
        if avg_latency > 300:
            new_size = min(self.current_buffer_size + 2, self.max_buffer_size)
            self.current_buffer_size = new_size
        elif avg_latency < 100:
            new_size = max(self.current_buffer_size - 1, self.min_buffer_size)
            self.current_buffer_size = new_size

    def get_buffer_size(self) -> int:
        return self.current_buffer_size

    def get_metrics(self) -> dict[str, float]:
        if not self.latency_samples:
            return {}
        return {
            "avg_latency_ms": sum(self.latency_samples) / len(self.latency_samples),
            "min_latency_ms": min(self.latency_samples),
            "max_latency_ms": max(self.latency_samples),
            "current_buffer_size": self.current_buffer_size,
            "sample_count": len(self.latency_samples),
        }


def get_streaming_handler() -> StreamingResponseHandler:
    return StreamingResponseHandler()


def get_adaptive_controller() -> AdaptiveStreamingController:
    return AdaptiveStreamingController()
