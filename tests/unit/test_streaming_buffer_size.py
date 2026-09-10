"""DEFECT-3 회귀 테스트: 콘텐츠 버퍼 크기가 실제 스트리밍 청크 수에 반영된다.

Plan `streaming-fix-plan-v2.md` DEFECT-3 Test Matrix row 2 (HIGH-1c, MED-8):

- `api.streaming_handler.UI_CONTENT_BUFFER_SIZE` 를 4 로 패치한 뒤
  `get_streaming_handler()` 로 생성한 핸들러는 `content_buffer_size=4` 의
  PriorityStreamBuffer 를 가진다 (buffer_size == 4).
- 17개 콘텐츠 토큰을 messages 모드로 소비하면 비어 있지 않은 콘텐츠 청크가
  정확히 5개 나온다 — 첫 토큰 TTFT 바이패스 1 + 크기 4 배치 4개, 꼬리 없음.
- 대조: 크기 1 이면 17개 청크.
- 첫 청크 내용은 첫 토큰과 같다 (바이패스 보존).
- 꼬리가 있는 입력(20 토큰)에서는 handler finally flush 가 is_final=True 인
  마지막 청크를 방출한다 (plan NOTE: 1 + ceil(19/4) = 6).

모듈 경로는 canonical 인 `api.streaming_handler` 를 사용한다 (conftest 가
src/ 를 sys.path 에 넣으므로 `src.api.streaming_handler` 는 별도 복제본이어서
패치 대상으로 쓰면 handler 에 영향이 없다 — test_stream_cancel.py 참조).
"""

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

import pytest

import api.streaming_handler as streaming_handler_mod
from api.streaming_handler import StreamChunk


class _FakeChunk:
    """messages 모드 청크 객체 — handler 가 hasattr(chunk_obj, 'content') 로 소비."""

    def __init__(self, content: str = "") -> None:
        self.content = content
        self.content_blocks: list[dict[str, object]] = []
        self.additional_kwargs: dict[str, object] = {}


async def _content_tokens(n_tokens: int) -> AsyncIterator[tuple[str, Any]]:
    """``stream_graph_events`` 가 소비하는 messages 모드 이벤트 스트림."""
    for i in range(n_tokens):
        yield ("messages", (_FakeChunk(content=f"tok{i} "), {}))


@pytest.mark.asyncio
async def test_buffer_size_reduces_chunk_count():
    """버퍼 크기 4: 17 토큰 → 정확히 5 청크. 크기 1 대조: 17 청크."""
    with patch.object(streaming_handler_mod, "UI_CONTENT_BUFFER_SIZE", 4):
        handler = streaming_handler_mod.get_streaming_handler()

        # 테스트 seam: get_streaming_handler() 가 call-time 으로 상수를 읽으므로
        # 버퍼 크기가 실제로 4 가 되어야 한다.
        assert handler.buffer.buffer_size == 4

        chunks: list[StreamChunk] = []
        async for chunk in handler.stream_graph_events(_content_tokens(17)):
            chunks.append(chunk)

        content_chunks = [c for c in chunks if c.content]
        assert len(content_chunks) == 5, (
            f"size=4: expected 5 content chunks (1 first-token bypass + 4 batches), "
            f"got {len(content_chunks)} from {[c.content for c in chunks]}"
        )
        assert all(c.content for c in content_chunks)
        # 첫 토큰 바이패스 보존: 첫 청크 내용은 정확히 첫 토큰.
        assert content_chunks[0].content == "tok0 "
        # 17 = 1 + 4*4 → 꼬리 없음. 마지막 청크는 배치 청크이며 is_final=False.
        assert content_chunks[-1].is_final is False

    with patch.object(streaming_handler_mod, "UI_CONTENT_BUFFER_SIZE", 1):
        baseline = streaming_handler_mod.get_streaming_handler()
        assert baseline.buffer.buffer_size == 1

        single_chunks: list[StreamChunk] = []
        async for chunk in baseline.stream_graph_events(_content_tokens(17)):
            single_chunks.append(chunk)

        assert len([c for c in single_chunks if c.content]) == 17, (
            "size=1: every token must flush immediately (17 chunks)"
        )


@pytest.mark.asyncio
async def test_buffer_size_tail_flush_is_final():
    """꼬리가 남는 입력(20 토큰)에서는 handler finally flush 가 is_final=True 청크를 만든다."""
    with patch.object(streaming_handler_mod, "UI_CONTENT_BUFFER_SIZE", 4):
        handler = streaming_handler_mod.get_streaming_handler()

        chunks: list[StreamChunk] = []
        async for chunk in handler.stream_graph_events(_content_tokens(20)):
            chunks.append(chunk)

        content_chunks = [c for c in chunks if c.content]
        # plan NOTE: 1 + ceil(19/4) = 6 (마지막 3-토큰 꼬리).
        assert len(content_chunks) == 6, (
            f"size=4, 20 tokens: expected 1 + ceil(19/4) = 6 chunks, "
            f"got {len(content_chunks)}"
        )
        assert content_chunks[0].content == "tok0 "
        assert content_chunks[-1].is_final is True
        assert content_chunks[-1].content == "tok17 tok18 tok19 "
