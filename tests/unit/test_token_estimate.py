"""T3 회귀 테스트: _estimate_tokens() 일관성 및 경계값 검증."""

from collections.abc import AsyncIterator
from typing import Any

import pytest

from api.streaming_handler import StreamingResponseHandler, _estimate_tokens


class _FakeChunk:
    """messages 모드 청크 — handler 가 hasattr(chunk_obj, 'content') 로 소비."""

    def __init__(self, content: str = "") -> None:
        self.content = content
        self.content_blocks: list[dict[str, object]] = []
        self.additional_kwargs: dict[str, object] = {}


def test_estimate_tokens_bounds() -> None:
    assert _estimate_tokens("") == 1
    assert _estimate_tokens("a") == 1
    assert _estimate_tokens("ab") == 1
    s = "korean문장입니"
    assert _estimate_tokens(s) == max(1, len(s) // 2)


@pytest.mark.asyncio
async def test_token_counts_consistent_across_paths() -> None:
    X = "hello world 테스트"

    async def custom_stream() -> AsyncIterator[tuple[str, Any]]:
        yield ("custom", {"content": X})

    async def messages_stream() -> AsyncIterator[tuple[str, Any]]:
        yield ("messages", _FakeChunk(content=X))

    handler_custom = StreamingResponseHandler()
    chunks_custom = [
        c async for c in handler_custom.stream_graph_events(custom_stream())
    ]

    handler_messages = StreamingResponseHandler()
    chunks_messages = [
        c async for c in handler_messages.stream_graph_events(messages_stream())
    ]

    custom_content = [c for c in chunks_custom if c.content]
    messages_content = [c for c in chunks_messages if c.content]

    assert len(custom_content) >= 1
    assert len(messages_content) >= 1
    assert custom_content[0].token_count == messages_content[0].token_count
