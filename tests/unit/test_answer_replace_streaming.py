"""Regenerated-answer replacement across the streaming consumers.

When a generation pass is retried for length, the 2nd pass is not streamed
(``pass_writer=None``); the authoritative body arrives as a ``replace`` event.
These tests pin the contract for every consumer:

- ``stream_graph_events`` turns the event into a replace chunk,
- ``_content_generator`` stores the replacement in aux state without
  re-yielding it (live text was already painted by the 1st pass),
- ``consume_stream_into_message`` persists the replacement,
- ``chunk_to_stream_events`` marks the message payload for SSE clients,
- ``_resolve_final_text`` prefers the replacement unless the turn errored.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("IS_CI_TEST", "true")

from api.stream_pipeline import (  # noqa: E402
    StreamChunk,
    StreamingResponseHandler,
    chunk_to_stream_events,
)
from core.rag_core import RAGSystem  # noqa: E402
from core.session import SessionManager  # noqa: E402
from ui.components.chat import _resolve_final_text  # noqa: E402
from ui.components.streaming import (  # noqa: E402
    _AUX_STATE_KEY,
    consume_stream_into_message,
)
from ui.components.streaming_core import _content_generator  # noqa: E402

_SHORT = "1차 패스의 짧은 답변입니다."
_FINAL = "재생성된 최종 답변입니다. " * 40


def test_stream_graph_events_replace_mode_emits_replace_chunk() -> None:
    citations = [{"doc_id": "doc-1", "page": 3}]

    async def _events():
        yield ("custom", {"content": _SHORT})
        yield ("replace", {"content": _FINAL, "citations": citations})

    async def _collect() -> list[StreamChunk]:
        handler = StreamingResponseHandler()
        return [chunk async for chunk in handler.stream_graph_events(_events())]

    chunks = asyncio.run(_collect())
    replaces = [chunk for chunk in chunks if chunk.is_replace]
    assert len(replaces) == 1
    assert replaces[0].content == _FINAL
    assert replaces[0].citations == citations
    assert replaces[0].is_final is True

    replace_index = chunks.index(replaces[0])
    assert all(not chunk.content for chunk in chunks[replace_index + 1 :])


def test_chunk_to_stream_events_marks_replace_message() -> None:
    events = chunk_to_stream_events(StreamChunk(content=_FINAL, is_replace=True))
    message = next(event for event in events if event.type == "message")
    assert message.payload == {"content": _FINAL, "replace": True}

    plain = chunk_to_stream_events(StreamChunk(content=_SHORT))
    plain_message = next(event for event in plain if event.type == "message")
    assert plain_message.payload == {"content": _SHORT}


def test_content_generator_replace_overrides_streamed_content() -> None:
    sid = "answer-replace-generator"
    SessionManager.reset_all_state(sid)
    chunks = [
        StreamChunk(content=_SHORT),
        StreamChunk(content=_FINAL, is_replace=True),
    ]

    import ui.components.streaming_core as streaming_core_mod

    with patch.object(streaming_core_mod, "stream_chunks", return_value=iter(chunks)):
        yielded = list(_content_generator("q", "m", sid, "mid"))

    assert "".join(yielded) == _SHORT
    aux = SessionManager.get(_AUX_STATE_KEY, {}, session_id=sid)
    assert aux["content"] == _FINAL
    assert aux["complete"] is True


def test_consume_stream_into_message_persists_replaced_content() -> None:
    sid = "answer-replace-consume"
    SessionManager.reset_all_state(sid)
    chunks = [
        StreamChunk(content=_SHORT),
        StreamChunk(content=_FINAL, is_replace=True),
    ]

    import ui.components.streaming as streaming_mod

    with patch.object(streaming_mod, "stream_chunks", return_value=iter(chunks)):
        message = consume_stream_into_message(
            sid, "질문", "test-model", msg_id="mid-replace"
        )

    assert message is not None
    assert message["content"] == _FINAL


@pytest.mark.asyncio
async def test_rag_core_astream_emits_replace_when_regenerated() -> None:
    sid = "answer-replace-astream"
    rag = RAGSystem(session_id=sid)
    engine = AsyncMock()
    breaker = MagicMock()
    registry = MagicMock()
    registry.get_breaker.return_value = breaker
    manager = MagicMock()
    manager.retrievers._pool = {}
    session_manager = MagicMock()
    session_manager.get.return_value = None
    session_manager.get_messages.return_value = []

    def _generate(response: str) -> dict:
        return {
            "event": "on_chain_stream",
            "metadata": {},
            "data": {"chunk": {"generate": {"response": response}}},
        }

    def _replace(content: str) -> dict:
        return {
            "event": "on_custom_event",
            "name": "answer_replace",
            "metadata": {},
            "data": {
                "content": content,
                "citations": [{"doc_id": "doc-1"}],
                "regenerated": "under_budget",
            },
        }

    async def fake_stream(*args, **kwargs):
        yield _generate(_SHORT)
        yield _replace(_FINAL)

    breaker.call_async_stream = lambda *args, **kwargs: fake_stream()

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
        stack.enter_context(patch("core.rag_core.record_qa_turn"))
        gen = await rag.astream("질문")
        received = [item async for item in gen]

    replaces = [item for item in received if item[0] == "replace"]
    assert replaces == [
        (
            "replace",
            {
                "content": _FINAL,
                "citations": [{"doc_id": "doc-1"}],
                "regenerated": "under_budget",
            },
        )
    ]


def test_resolve_final_text_prefers_replacement_and_keeps_errors() -> None:
    assert _resolve_final_text(_FINAL, _SHORT, None) == (_FINAL, True)
    assert _resolve_final_text(_SHORT, _SHORT, None) == (_SHORT, False)
    assert _resolve_final_text(_FINAL, "friendly error", "boom") == (
        "friendly error",
        False,
    )
