"""
스트리밍 상태 머신 모듈 - Phase 2.5 상태 전이 관리
Task 12의 streaming_handler.py에서 분리 (PHASE 3-P1)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from api.stream_buffers import PriorityStreamBuffer
    from api.streaming_handler import StreamChunk

logger = logging.getLogger(__name__)

__all__ = [
    "StreamingState",
    "StreamingStateContext",
    "StreamingStateMachine",
    "create_streaming_state_machine",
]


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
        import time

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
