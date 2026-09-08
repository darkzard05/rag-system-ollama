"""
스트리밍 버퍼 모듈 - 토큰/우선순위 이중 버퍼링
Task 12의 streaming_handler.py에서 분리 (PHASE 3-P1)
"""

import time

__all__ = ["PriorityStreamBuffer", "TokenStreamBuffer"]


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
        # Content 버퍼: 즉시 플러시 (size=1, timeout=10ms)
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
