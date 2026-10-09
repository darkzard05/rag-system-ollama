"""Phase 1 통합: citations_add 디스패치 → 파이프라인 → UI 병합.

- backend custom event → StreamChunk (stream_graph_events)
- StreamChunk → StreamEvent (chunk_to_stream_events)
- UI 병합 (_content_generator) + prewarm 제출 상한(<=3)
- rerun-skip 가드 경로: clean turn에서 증분 인용이 페인트되고
  ``stream_finalized_no_rerun`` 이 설정된다.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import ui.components.chat as chat_module
from api.stream_pipeline import (
    StreamChunk,
    StreamingResponseHandler,
    chunk_to_stream_events,
)
from core.session import SessionManager
from ui.components import streaming_core
from ui.components.streaming_core import _content_generator

_SID = "incr-dispatch-sid"
_MSG = "incr-dispatch-msg"


def _cit(i: int) -> dict:
    return {
        "doc_id": f"doc-{i}",
        "page": i + 1,
        "section": f"s{i}",
        "text_span": f"t{i}",
    }


def _add_payload(
    items: list[dict], base_index: int, arr_len: int, attempt: int
) -> dict:
    return {
        "items": items,
        "base_index": base_index,
        "arr_len": arr_len,
        "attempt": attempt,
    }


@pytest.mark.asyncio
async def test_backend_custom_event_becomes_stream_chunk() -> None:
    """backend ``citations_add`` custom event → StreamChunk.citations_add."""

    async def _stream() -> AsyncIterator[tuple[str, Any]]:
        yield ("custom", _add_payload([_cit(0)], 0, 1, 0))

    handler = StreamingResponseHandler()
    chunks = [c async for c in handler.stream_graph_events(_stream())]
    add_chunks = [c for c in chunks if c.citations_add is not None]
    assert len(add_chunks) == 1
    assert add_chunks[0].citations_add == _add_payload([_cit(0)], 0, 1, 0)


def test_chunk_to_stream_events_preserves_citations_add_payload() -> None:
    chunk = StreamChunk(citations_add=_add_payload([_cit(0), _cit(1)], 0, 2, 0))
    events = chunk_to_stream_events(chunk)
    assert len(events) == 1
    assert events[0].type == "citations_add"
    assert events[0].payload == _add_payload([_cit(0), _cit(1)], 0, 2, 0)


def test_ui_merges_incremental_citations_and_bounds_prewarm() -> None:
    """fake backend chunks → _content_generator 병합 + prewarm <= 3."""
    SessionManager.reset_all_state(_SID)
    chunks = [
        StreamChunk(content="tok"),
        StreamChunk(citations_add=_add_payload([_cit(0)], 0, 1, 0)),
        StreamChunk(citations_add=_add_payload([_cit(1)], 1, 2, 0)),
        StreamChunk(citations_add=_add_payload([_cit(2)], 2, 3, 0)),
        StreamChunk(citations=[_cit(0), _cit(1), _cit(2)]),
    ]
    prewarm_calls: list[tuple] = []

    def _fake_prewarm(sid, msg_id, documents, citations):
        prewarm_calls.append((sid, msg_id, list(documents), list(citations)))

    with (
        patch.object(streaming_core, "stream_chunks", return_value=iter(chunks)),
        patch.object(
            streaming_core, "_note_reference_prewarm", side_effect=_fake_prewarm
        ),
    ):
        list(_content_generator("q", "m", _SID, _MSG))

    final = SessionManager.get("stream_active_aux", None, _SID)
    assert final is not None
    assert final["citations"] == [_cit(0), _cit(1), _cit(2)]
    # first_add + final = 2 (5s 미경과이므로 elapsed 없음)
    assert len(prewarm_calls) <= 3


def _make_st_mock() -> MagicMock:
    st_mock = MagicMock(name="st")
    st_mock.session_state = {}
    st_mock.write_stream.side_effect = lambda gen: "".join(gen)
    return st_mock


def test_clean_turn_paints_incremental_citations_and_skips_rerun() -> None:
    """rerun-skip 가드 경로: clean turn에서 증분 인용 페인트 + no rerun."""
    sid = f"incr_{uuid.uuid4().hex[:8]}"
    mid = f"msg_{uuid.uuid4().hex[:8]}"
    SessionManager.reset_all_state(sid)
    st_mock = _make_st_mock()
    chunks = [
        StreamChunk(content="hello "),
        StreamChunk(citations_add=_add_payload([_cit(0)], 0, 1, 0)),
        StreamChunk(content="world"),
        StreamChunk(citations=[_cit(0)]),
    ]
    with (
        patch.object(chat_module, "st", st_mock),
        patch.object(streaming_core, "stream_chunks", return_value=iter(chunks)),
        patch.object(chat_module, "_finalize_pdf_side_effects", MagicMock()),
        patch.object(chat_module, "render_generation_expander", MagicMock()),
        patch.object(
            chat_module, "render_inline_citation_badges", MagicMock(return_value=True)
        ),
        patch.object(chat_module, "_render_references_content", MagicMock()),
        patch.object(streaming_core, "_note_reference_prewarm", MagicMock()),
    ):
        chat_module._render_streaming_with_write_stream(
            {"msg_id": mid, "query": "q", "content": ""}, sid, "q"
        )

    st_mock.rerun.assert_not_called()
    assert SessionManager.get("stream_finalized_no_rerun", None, sid) == mid
    messages = SessionManager.get_messages(session_id=sid)
    assert messages, "assistant 메시지가 영속화되어야 한다"
    assert messages[-1]["citations"] == [_cit(0)]
    assert messages[-1]["content"] == "hello world"


def test_clean_turn_paints_badges_and_metrics_below_body_in_run() -> None:
    """실앱 재현: rerun 없이 본문 아래에 참조 목록이 그려져야 한다 (단순안).

    호출 순서(body → refs)를 공유 기록으로 고정한다.
    """
    sid = f"below_{uuid.uuid4().hex[:8]}"
    mid = f"msg_{uuid.uuid4().hex[:8]}"
    SessionManager.reset_all_state(sid)
    order: list[str] = []
    st_mock = _make_st_mock()

    def _write_stream(gen):
        text = "".join(gen)
        order.append("body")
        return text

    st_mock.write_stream.side_effect = _write_stream
    refs_mock = MagicMock(
        name="refs",
        return_value=True,
        side_effect=lambda *a, **k: order.append("refs"),
    )
    chunks = [
        StreamChunk(content="hello "),
        StreamChunk(citations_add=_add_payload([_cit(0)], 0, 1, 0)),
        StreamChunk(content="world"),
        StreamChunk(
            citations=[_cit(0)],
            performance={
                "total_time": 5.0,
                "input_token_count": 10,
                "token_count": 20,
            },
        ),
    ]
    with (
        patch.object(chat_module, "st", st_mock),
        patch.object(streaming_core, "stream_chunks", return_value=iter(chunks)),
        patch.object(chat_module, "_finalize_pdf_side_effects", MagicMock()),
        patch.object(chat_module, "render_reference_list", refs_mock),
        patch.object(streaming_core, "_note_reference_prewarm", MagicMock()),
    ):
        chat_module._render_streaming_with_write_stream(
            {"msg_id": mid, "query": "q", "content": ""}, sid, "q"
        )

    st_mock.rerun.assert_not_called()
    # 확정 조건은 꼬리 순서다: 본문 완료(body) 뒤에 참조 목록이 같은 run에서
    # 이어져야 한다 (배지/지표 없음).
    assert order[-2:] == ["body", "refs"], f"paint order: {order}"
    assert refs_mock.call_count >= 1
