"""P2 회귀 테스트: `_generate_pass`는 response_chunk를 합쳐서 dispatch한다.

계약:
- 17개 토큰 스트림에서 response_chunk dispatch는 토큰 수보다 적다
  (첫 토큰 즉시 1 + 배치 flush, 꼬리 flush 포함).
- 합친 content를 이어붙이면 원본 JSON과 동일하다 (무손실).
- 첫 dispatch의 content는 첫 토큰과 같다 (TTFT 보존).
- 꼬리 citation도 누락 없이 citations_add로 방출된다.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import core.graph._generate as gen


class _FakeChunk:
    """astream 최소 청크 객체 (fallback 경로: content 속성 str)."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.response_metadata: dict[str, Any] = {}


def _make_llm(tokens: list[str]) -> MagicMock:
    chunks = [_FakeChunk(t) for t in tokens]

    async def _astream(_messages: Any, config: Any = None) -> Any:
        for c in chunks:
            yield c

    llm = MagicMock()
    llm.bind.return_value.astream = _astream
    # hasattr(llm, "_convert...")가 False가 되도록 spec 없는 MagicMock 속성 삭제
    del llm._convert_chunk_to_thought_and_content
    return llm


async def _aenter(*args: Any, **kwargs: Any) -> Any:
    return None


async def _aexit(*args: Any, **kwargs: Any) -> Any:
    return False


def _split(text: str, n: int) -> list[str]:
    return [text[i : i + n] for i in range(0, len(text), n)]


PAYLOAD = (
    '{"final_answer":"stream coalescing keeps every byte",'
    '"citations":['
    '{"doc_id":"d1","text_span":"span one","section":"s1","page":1,"score":0.9},'
    '{"doc_id":"d2","text_span":"span two","section":"s2","page":2,"score":0.8}'
    "]}"
)


@patch.object(gen, "get_resource_manager")
@patch.object(gen.ModelManager, "inference_session")
@pytest.mark.asyncio
async def test_response_chunks_are_coalesced_losslessly(
    inference_session: MagicMock, resource_manager: MagicMock
) -> None:
    inference_session.return_value.__aenter__ = _aenter
    inference_session.return_value.__aexit__ = _aexit
    coordinator = resource_manager.return_value
    coordinator.use_llm.return_value.__aenter__ = _aenter
    coordinator.use_llm.return_value.__aexit__ = _aexit

    tokens = _split(PAYLOAD, 5)
    assert len(tokens) > 10  # 배치가 의미를 가지려면 토큰이 충분히 많아야 함
    llm = _make_llm(tokens)
    events: list[tuple[str, dict[str, Any]]] = []

    async def _capture(
        name: str,
        data: dict[str, Any],
        *,
        writer: Any,
        config: Any,
    ) -> None:
        events.append((name, data))

    with patch.object(gen, "_dispatch_event", _capture):
        await gen._generate_pass(
            llm=llm,
            sys_msg=MagicMock(),
            human_msg=MagicMock(),
            prompt_version="v1",
            config={},
            writer=object(),
            attempt=1,
        )

    chunks = [d for n, d in events if n == "response_chunk"]
    assert 1 < len(chunks) < len(tokens), (
        f"expected coalesced dispatch (1 < n < {len(tokens)}), got {len(chunks)}"
    )
    # 무손실: 합치면 원본과 동일
    assert "".join(c["content"] for c in chunks) == PAYLOAD
    # TTFT 보존: 첫 dispatch는 첫 토큰 그 자체
    assert chunks[0]["content"] == tokens[0]
    # 꼬리 citation 누락 금지 (증분 방출이므로 전 이벤트를 합쳐서 판정)
    cites = [d for n, d in events if n == "citations_add"]
    assert cites, "trailing citations must still be dispatched"
    streamed_ids = [o["doc_id"] for d in cites for o in d["items"]]
    assert streamed_ids == ["d1", "d2"], f"citation loss: {streamed_ids}"


@patch.object(gen, "get_resource_manager")
@patch.object(gen.ModelManager, "inference_session")
@pytest.mark.asyncio
async def test_writer_none_stays_silent(
    inference_session: MagicMock, resource_manager: MagicMock
) -> None:
    """writer=None 경로는 dispatch 없이 파싱만 수행한다 (기존 계약 유지)."""
    inference_session.return_value.__aenter__ = _aenter
    inference_session.return_value.__aexit__ = _aexit
    coordinator = resource_manager.return_value
    coordinator.use_llm.return_value.__aenter__ = _aenter
    coordinator.use_llm.return_value.__aexit__ = _aexit

    llm = _make_llm(_split(PAYLOAD, 5))
    events: list[tuple[str, dict[str, Any]]] = []

    async def _capture(
        name: str,
        data: dict[str, Any],
        *,
        writer: Any,
        config: Any,
    ) -> None:
        events.append((name, data))

    with patch.object(gen, "_dispatch_event", _capture):
        result = await gen._generate_pass(
            llm=llm,
            sys_msg=MagicMock(),
            human_msg=MagicMock(),
            prompt_version="v1",
            config={},
            writer=None,
            attempt=1,
        )

    assert events == []
    assert result["parse_failed"] is False
