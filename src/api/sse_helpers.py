"""
SSE 헬퍼 모듈 - SSE 이벤트 직렬화 및 응답 버퍼
Task 12의 streaming_handler.py에서 분리 (PHASE 3-P1)
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

import orjson

if TYPE_CHECKING:
    from api.stream_pipeline import StreamChunk

__all__ = [
    "ServerSentEventsHandler",
    "StreamingResponseBuilder",
    "gzip_compress",
]


def gzip_compress(data: str) -> bytes:
    """SSE 데이터 gzip 압축 (압축 임계값은: 1KB)"""
    import gzip

    if len(data) < 1024:
        return data.encode("utf-8")
    return gzip.compress(data.encode("utf-8"))


class ServerSentEventsHandler:
    @staticmethod
    def format_sse_event(
        event_type: str | None, data: dict[str, Any], event_id: int | None = None
    ) -> str:
        id_part = f"id: {event_id}\n" if event_id is not None else ""
        event_part = f"event: {event_type}\n" if event_type else ""
        return f"{id_part}{event_part}data: {orjson.dumps(data).decode('utf-8')}\n\n"

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

    @staticmethod
    def format_sse_batch(
        events: list[tuple[str | None, dict[str, Any], int | None]],
    ) -> str:
        """
        여러 SSE 이벤트를 배치로 직렬화합니다.
        Phase 2.4: 네트워크 라운드트립 감소용 배치 포장
        """
        if not events:
            return ""

        return "".join(
            ServerSentEventsHandler.format_sse_event(event_type, data, event_id)
            for event_type, data, event_id in events
        )

    @staticmethod
    def gzip_compress(data: str) -> bytes:
        return gzip_compress(data)


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
