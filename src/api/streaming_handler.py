"""
스트리밍 응답 처리 - Task 12
실시간 토큰 스트리밍, SSE 지원, UI 업데이트 최적화
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Coroutine, Dict, List, Optional
from datetime import datetime
import time

from services.monitoring.performance_monitor import (
    get_performance_monitor,
    OperationType,
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
    thought: Optional[str] = None  # 사고 과정 필드 추가


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
        self.buffer: List[str] = []
        self.last_flush_time: float = time.time()

    def add_token(self, token: str) -> Optional[str]:
        """
        토큰 추가

        Returns:
            플러시해야 할 버퍼 내용, 또는 None
        """
        self.buffer.append(token)

        current_time = time.time()
        elapsed_ms = (current_time - self.last_flush_time) * 1000

        # 버퍼 풀 또는 타임아웃 시 플러시
        if len(self.buffer) >= self.buffer_size or elapsed_ms >= self.timeout_ms:
            return self.flush()

        return None

    def flush(self) -> Optional[str]:
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
        self.start_time: Optional[float] = None
        self.first_token_time: Optional[float] = None

    async def stream_response(
        self,
        response_generator: AsyncIterator[str],
        on_chunk: Callable[[StreamChunk], Coroutine[Any, Any, None]],
        on_complete: Optional[Callable[[], Coroutine[Any, Any, None]]] = None,
        on_error: Optional[Callable[[Exception], Coroutine[Any, Any, None]]] = None,
        operation_name: str = "response_streaming",
    ) -> StreamingMetrics:
        """
        응답을 스트리밍으로 처리

        Args:
            response_generator: 토큰을 생성하는 비동기 이터레이터
            on_chunk: 청크가 도착할 때 호출할 콜백
            on_complete: 스트리밍 완료 시 호출할 콜백
            on_error: 에러 발생 시 호출할 콜백
            operation_name: 작업 이름

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
        event_type: str, data: Dict[str, Any], event_id: Optional[int] = None
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
        self.chunks: List[StreamChunk] = []
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

    def get_chunks(self) -> List[StreamChunk]:
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
        initial_buffer_size: int = 10,
        min_buffer_size: int = 5,
        max_buffer_size: int = 50,
    ):
        self.current_buffer_size = initial_buffer_size
        self.min_buffer_size = min_buffer_size
        self.max_buffer_size = max_buffer_size
        self.latency_samples: List[float] = []
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

    def get_metrics(self) -> Dict[str, float]:
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
_streaming_handler: Optional[StreamingResponseHandler] = None
_adaptive_controller: Optional[AdaptiveStreamingController] = None


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
