"""RAGSystem.astream 내부 제어 노드 토큰 필터링 검증.

`grade_documents` / `verify_answer` 노드는 내부 라우팅·검증용 JSON을 생성하는
제어 노드이며 사용자 응답이 아니다. LangGraph `astream_events(version="v2")`
에서는 이 노드들의 `on_chat_model_stream` 이벤트도 전파되므로, `_consumer`가
이를 드롭하지 않으면 내부 JSON 토큰이 사용자 응답으로 유출된다.

또한 `dropped_generate_msgs`는 generate 노드 전용 계측 카운터이므로, 제어
노드 드롭 시에는 증가하지 않아야 한다(무응답 오탐 경고 방지).
"""

from collections.abc import Iterator
from contextlib import ExitStack
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessageChunk

from core.rag_core import RAGSystem


def _stream(events: list[dict[str, Any]]):
    async def fake_stream(*args: Any, **kwargs: Any):
        for event in events:
            yield event

    return fake_stream


def _chat_stream(node: str | None, content: str) -> dict[str, Any]:
    return {
        "event": "on_chat_model_stream",
        "metadata": {"langgraph_node": node},
        "data": {"chunk": AIMessageChunk(content=content)},
    }


def _custom_response(content: str) -> dict[str, Any]:
    return {
        "event": "on_custom_event",
        "name": "response_chunk",
        "metadata": {"langgraph_node": "generate"},
        "data": {"content": content},
    }


@pytest.fixture
def astream_env() -> Iterator[dict[str, Any]]:
    rag = RAGSystem(session_id="test_rag_astream_node_filtering")
    engine = AsyncMock()
    breaker = MagicMock()
    registry = MagicMock()
    registry.get_breaker.return_value = breaker
    manager = MagicMock()
    manager.retrievers._pool = {"doc1": "r1"}
    session_manager = MagicMock()
    session_manager.get.return_value = None

    env: dict[str, Any] = {"rag": rag, "breaker": breaker, "manager": manager}

    with ExitStack() as stack:
        stack.enter_context(patch("core.rag_core.SessionManager", session_manager))
        stack.enter_context(
            patch(
                "core.rag_core.prepare_query_config_or_build",
                AsyncMock(return_value={}),
            )
        )
        stack.enter_context(
            patch.object(rag, "_get_rag_engine", AsyncMock(return_value=engine))
        )
        stack.enter_context(
            patch("core.rag_core.get_circuit_breaker_registry", return_value=registry)
        )
        stack.enter_context(
            patch("core.rag_core.get_resource_manager", return_value=manager)
        )
        stack.enter_context(patch("core.rag_core.hydrate_documents", AsyncMock()))
        yield env


async def _collect(rag: RAGSystem) -> list[tuple[str, Any]]:
    gen = await rag.astream("질문")
    out: list[tuple[str, Any]] = []
    async for item in gen:
        out.append(item)
    return out


@pytest.mark.asyncio
async def test_generate_custom_and_unattributed_stream_pass_through(
    astream_env: dict[str, Any],
) -> None:
    """generate custom 이벤트와 노드 미지정 chat 스트림은 그대로 전달된다."""
    astream_env["breaker"].call_async_stream = _stream(
        [
            _custom_response("최종 답변입니다."),
            _chat_stream(None, "노드 미지정 청크"),
        ]
    )

    result = await _collect(astream_env["rag"])

    assert ("custom", {"content": "최종 답변입니다."}) in result
    assert any(kind == "messages" for kind, _ in result)


@pytest.mark.asyncio
async def test_control_node_json_tokens_are_not_leaked(
    astream_env: dict[str, Any],
) -> None:
    """grade_documents / verify_answer JSON 토큰은 스트림에 유출되지 않는다."""
    astream_env["breaker"].call_async_stream = _stream(
        [
            _chat_stream("grade_documents", '{"action": "generate"}'),
            _custom_response("최종 답변입니다."),
            _chat_stream("verify_answer", '{"faithful": true}'),
        ]
    )

    result = await _collect(astream_env["rag"])

    assert result == [("custom", {"content": "최종 답변입니다."})]


@pytest.mark.asyncio
async def test_control_node_drop_does_not_trigger_missing_answer_warning(
    astream_env: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """제어 노드 토큰만 드롭된 경우 generate 무응답 경고가 없어야 한다."""
    astream_env["breaker"].call_async_stream = _stream(
        [_chat_stream("grade_documents", '{"action": "generate"}')]
    )

    with caplog.at_level("WARNING", logger="core.rag_core"):
        result = await _collect(astream_env["rag"])

    assert result == []
    assert "generate 토큰 드롭됨" not in caplog.text


@pytest.mark.asyncio
async def test_dropped_generate_tokens_still_warn_when_no_custom(
    astream_env: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """generate 토큰이 드롭되고 custom 이벤트가 없으면 기존 경고가 유지된다."""
    astream_env["breaker"].call_async_stream = _stream(
        [_chat_stream("generate", "중복 토큰")]
    )

    with caplog.at_level("WARNING", logger="core.rag_core"):
        result = await _collect(astream_env["rag"])

    assert result == []
    assert "generate 토큰 드롭됨" in caplog.text
