"""
스트리밍 응답 처리 - Task 12
실시간 토큰 스트리밍, SSE 지원, UI 업데이트 최적화
"""

import json
import logging
import time
from collections.abc import AsyncIterator, Callable, Coroutine
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from services.monitoring.performance_monitor import (
    OperationType,
    get_performance_monitor,
)

logger = logging.getLogger(__name__)
monitor = get_performance_monitor()


@dataclass
class StreamChunk:
    """스트리밍 청크 정보"""

    content: str
    timestamp: float
    token_count: int
    chunk_index: int
    is_final: bool = False
    thought: str | None = None  # 사고 과정 필드 추가
    metadata: dict[str, Any] | None = None  # 메타데이터 (문서 등) 추가


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

    특징:
    - 동적 버퍼 크기 조정
    - 타임아웃 기반 플러시
    - 토큰 카운팅
    """

    def __init__(self, buffer_size: int = 10, timeout_ms: float = 100.0):
        self.buffer_size = buffer_size
        self.timeout_ms = timeout_ms
        self.buffer: list[str] = []
        self.last_flush_time: float = time.time()

    def add_token(self, token: str) -> str | None:
        """
        토큰 추가
        """
        self.buffer.append(token)

        # 버퍼 풀 시 즉시 플러시 (시간 측정 없이)
        if len(self.buffer) >= self.buffer_size:
            return self.flush()

        # 타임아웃 기반 플러시는 일정 시간 간격으로만 체크 (선택적 최적화 가능)
        # 여기서는 오버헤드 감소를 위해 버퍼가 찰 때까지 기다리는 것을 우선함
        return None

    def flush(self) -> str | None:
        """버퍼 플러시"""
        if not self.buffer:
            return None

        content = "".join(self.buffer)
        self.buffer.clear()
        self.last_flush_time = time.time()
        return content

    def reset(self) -> None:
        """버퍼 초기화"""
        self.buffer.clear()
        self.last_flush_time = time.time()


class StreamingResponseHandler:
    """
    스트리밍 응답 처리기 - 실시간 토큰 스트리밍

    특징:
    - 토큰 단위 스트리밍
    - 성능 메트릭 수집
    - 에러 처리 및 복구
    - SSE 호환성
    """

    def __init__(self, buffer_size: int = 10, timeout_ms: float = 100.0):
        self.buffer = TokenStreamBuffer(buffer_size, timeout_ms)
        self.metrics = StreamingMetrics()
        self.chunk_index = 0
        self.start_time: float | None = None
        self.first_token_time: float | None = None

    async def stream_graph_events(
        self,
        event_stream: AsyncIterator[dict[str, Any]],
        adaptive_controller: "AdaptiveStreamingController | None" = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        LangGraph 이벤트를 소비하여 가공된 스트리밍 청크를 생성 (리소스 안전 관리 최적화)
        """
        from contextlib import aclosing

        self.start_time = time.time()
        self.chunk_index = 0
        self.metrics = StreamingMetrics()
        self.first_token_time = None
        self.buffer.reset()

        try:
            async with aclosing(event_stream) as stream:
                async for event in stream:
                    kind = event["event"]
                    name = event.get("name", "Unknown")
                    data = event.get("data", {})

                    # 1. 커스텀 응답 청크 이벤트 처리
                    if kind == "on_custom_event" and name == "response_chunk":
                        content = data.get("chunk", "")
                        thought = data.get("thought", "")
                        current_time = time.time()

                        # 사고 과정 처리 (버퍼링 없이 즉시 전송)
                        if thought:
                            yield StreamChunk(
                                content="",
                                timestamp=current_time,
                                token_count=0,
                                chunk_index=self.chunk_index,
                                thought=thought,
                            )
                            self.chunk_index += 1

                        # 답변 본문 처리 (적응형 버퍼링)
                        if content:
                            if self.first_token_time is None:
                                self.first_token_time = current_time

                            buffered_content = self.buffer.add_token(content)
                            if buffered_content:
                                if adaptive_controller:
                                    self.buffer.buffer_size = (
                                        adaptive_controller.get_buffer_size()
                                    )

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

                    # 2. 메타데이터 처리
                    elif kind == "on_chain_end":
                        if name == "retrieve":
                            output = data.get("output", {})
                            if "documents" in output:
                                yield StreamChunk(
                                    content="",
                                    timestamp=time.time(),
                                    token_count=0,
                                    chunk_index=self.chunk_index,
                                    metadata={"documents": output["documents"]},
                                )
                                self.chunk_index += 1
        except Exception as e:
            logger.error(f"[Streaming] 스트림 처리 중 오류: {e}")
            # 에러 발생 시 현재까지의 내용이라도 보내기 위해 아래 finally 절로 이동
        finally:
            # 남은 버퍼 플러시
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

            # 최종 메트릭 계산
            self.metrics.total_time = time.time() - self.start_time
            if self.first_token_time:
                self.metrics.first_token_latency = (
                    self.first_token_time - self.start_time
                )
            if self.metrics.total_time > 0:
                self.metrics.tokens_per_second = (
                    self.metrics.total_tokens / self.metrics.total_time
                )

    async def stream_response(
        self,
        response_generator: AsyncIterator[str],
        on_chunk: Callable[[StreamChunk], Coroutine[Any, Any, None]],
        on_complete: Callable[[], Coroutine[Any, Any, None]] | None = None,
        on_error: Callable[[Exception], Coroutine[Any, Any, None]] | None = None,
        operation_name: str = "response_streaming",
        adaptive_controller: "AdaptiveStreamingController | None" = None,
    ) -> StreamingMetrics:
        """
        응답을 스트리밍으로 처리

        Args:
            response_generator: 토큰을 생성하는 비동기 이터레이터
            on_chunk: 청크가 도착할 때 호출할 콜백
            on_complete: 스트리밍 완료 시 호출할 콜백
            on_error: 에러 발생 시 호출할 콜백
            operation_name: 작업 이름
            adaptive_controller: 적응형 스트리밍 제어기

        Returns:
            스트리밍 성능 메트릭
        """
        self.start_time = time.time()
        self.metrics = StreamingMetrics()
        self.chunk_index = 0

        with monitor.track_operation(
            OperationType.LLM_INFERENCE,
            {"stage": "streaming", "buffer_size": self.buffer.buffer_size},
        ) as op:
            try:
                async for token in response_generator:
                    # 적응형 버퍼 크기 적용
                    if adaptive_controller:
                        new_size = adaptive_controller.get_buffer_size()
                        if self.buffer.buffer_size != new_size:
                            self.buffer.buffer_size = new_size

                    # 첫 토큰 시간 기록
                    if self.first_token_time is None:
                        self.first_token_time = time.time()
                        self.metrics.first_token_latency = (
                            self.first_token_time - self.start_time
                        )
                        logger.info(
                            f"[Streaming] 첫 토큰 지연: {self.metrics.first_token_latency * 1000:.2f}ms"
                        )

                    # 버퍼에 토큰 추가
                    buffered_content = self.buffer.add_token(token)

                    if buffered_content:
                        # 청크 생성 및 전송
                        chunk = StreamChunk(
                            content=buffered_content,
                            timestamp=time.time(),
                            token_count=len(buffered_content.split()),
                            chunk_index=self.chunk_index,
                            is_final=False,
                        )

                        # 메트릭 업데이트
                        self.metrics.total_tokens += chunk.token_count
                        self.metrics.chunk_count += 1

                        # 지연 시간 추적
                        latency = chunk.timestamp - self.start_time
                        self.metrics.min_latency = min(
                            self.metrics.min_latency, latency
                        )
                        self.metrics.max_latency = max(
                            self.metrics.max_latency, latency
                        )

                        await on_chunk(chunk)
                        self.chunk_index += 1

                # 남은 버퍼 플러시
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

                # 성능 메트릭 최종 계산
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

                # 완료 콜백
                if on_complete:
                    await on_complete()

                logger.info(
                    f"[Streaming] 완료: "
                    f"{self.metrics.total_tokens} 토큰, "
                    f"{self.metrics.total_time:.2f}초, "
                    f"{self.metrics.tokens_per_second:.1f} tok/s"
                )

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
    """
    Server-Sent Events (SSE) 처리기

    특징:
    - SSE 형식 생성
    - Keep-alive 지원
    - 타임아웃 관리
    """

    @staticmethod
    def format_sse_event(
        event_type: str, data: dict[str, Any], event_id: int | None = None
    ) -> str:
        """
        SSE 형식으로 이벤트 포매팅

        Returns:
            SSE 형식 문자열
        """
        lines = []

        if event_id is not None:
            lines.append(f"id: {event_id}")

        lines.append(f"event: {event_type}")
        lines.append(f"data: {json.dumps(data, ensure_ascii=False)}")
        lines.append("")  # 빈 줄로 이벤트 종료

        return "\n".join(lines)

    @staticmethod
    def format_sse_error(error_message: str, error_code: int = 500) -> str:
        """SSE 에러 포매팅"""
        data = {
            "error": error_message,
            "code": error_code,
            "timestamp": datetime.now().isoformat(),
        }
        return ServerSentEventsHandler.format_sse_event("error", data)

    @staticmethod
    def format_sse_keepalive(message: str = "keep-alive") -> str:
        """SSE keep-alive 포매팅"""
        return f": {message}\n"


class StreamingResponseBuilder:
    """
    스트리밍 응답 빌더 - 청크를 누적하여 최종 응답 생성
    """

    def __init__(self, max_buffer_size: int = 100000):
        self.chunks: list[StreamChunk] = []
        self.max_buffer_size = max_buffer_size
        self.total_content = ""

    def add_chunk(self, chunk: StreamChunk) -> None:
        """청크 추가"""
        if len(self.total_content) + len(chunk.content) > self.max_buffer_size:
            logger.warning("[StreamingBuilder] 버퍼 크기 초과, 최신 청크부터 보관")
            # 오래된 청크 제거
            while self.chunks and len(self.total_content) > self.max_buffer_size * 0.8:
                removed = self.chunks.pop(0)
                self.total_content = self.total_content[len(removed.content) :]

        self.chunks.append(chunk)
        self.total_content += chunk.content

    def get_content(self) -> str:
        """누적된 전체 내용 반환"""
        return self.total_content

    def get_chunks(self) -> list[StreamChunk]:
        """모든 청크 반환"""
        return self.chunks

    def reset(self) -> None:
        """빌더 초기화"""
        self.chunks.clear()
        self.total_content = ""


class AdaptiveStreamingController:
    """
    적응형 스트리밍 제어기 - 네트워크 상태에 따라 버퍼 크기 자동 조정

    특징:
    - 네트워크 지연 감지
    - 동적 버퍼 크기 조정
    - 처리량 최적화
    """

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
        """지연 시간 기록"""
        self.latency_samples.append(latency_ms)

        # 샘플 유지
        if len(self.latency_samples) > self.max_samples:
            self.latency_samples.pop(0)

        # 버퍼 크기 조정
        self._adjust_buffer_size()

    def _adjust_buffer_size(self) -> None:
        """지연 시간 기반 버퍼 크기 조정"""
        if len(self.latency_samples) < 10:
            return

        avg_latency = sum(self.latency_samples) / len(self.latency_samples)

        # 지연이 높으면 버퍼 크기 증가 (배치 처리로 횟수 감소)
        if avg_latency > 200:  # 200ms 이상
            new_size = min(self.current_buffer_size + 5, self.max_buffer_size)
            if new_size != self.current_buffer_size:
                logger.info(
                    f"[AdaptiveStreaming] 지연 높음 ({avg_latency:.1f}ms), "
                    f"버퍼 증가: {self.current_buffer_size} → {new_size}"
                )
                self.current_buffer_size = new_size

        # 지연이 낮으면 버퍼 크기 감소 (더 빈번한 업데이트)
        elif avg_latency < 50:  # 50ms 이하
            new_size = max(self.current_buffer_size - 2, self.min_buffer_size)
            if new_size != self.current_buffer_size:
                logger.info(
                    f"[AdaptiveStreaming] 지연 낮음 ({avg_latency:.1f}ms), "
                    f"버퍼 감소: {self.current_buffer_size} → {new_size}"
                )
                self.current_buffer_size = new_size

    def get_buffer_size(self) -> int:
        """현재 버퍼 크기 반환"""
        return self.current_buffer_size

    def get_metrics(self) -> dict[str, float]:
        """성능 메트릭 반환"""
        if not self.latency_samples:
            return {}

        return {
            "avg_latency_ms": sum(self.latency_samples) / len(self.latency_samples),
            "min_latency_ms": min(self.latency_samples),
            "max_latency_ms": max(self.latency_samples),
            "current_buffer_size": self.current_buffer_size,
            "sample_count": len(self.latency_samples),
        }


# 전역 핸들러 인스턴스
_streaming_handler: StreamingResponseHandler | None = None
_adaptive_controller: AdaptiveStreamingController | None = None


def get_streaming_handler() -> StreamingResponseHandler:
    """스트리밍 응답 처리기 인스턴스 반환"""
    global _streaming_handler
    if _streaming_handler is None:
        _streaming_handler = StreamingResponseHandler()
    return _streaming_handler


def get_adaptive_controller() -> AdaptiveStreamingController:
    """적응형 스트리밍 제어기 인스턴스 반환"""
    global _adaptive_controller
    if _adaptive_controller is None:
        _adaptive_controller = AdaptiveStreamingController()
    return _adaptive_controller


def reset_streaming_handler() -> None:
    """스트리밍 처리기 초기화"""
    global _streaming_handler
    _streaming_handler = None
