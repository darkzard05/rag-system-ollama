"""스트리밍 이벤트 매핑 계약(contract) 테스트.

``api.stream_events`` 가 단일 진실 소스(SSoT)로 보장하는 여섯 가지 이벤트
타입과 canonical 순서가 두 소비 경로에 동일하게 전달되는지 검증한다:

- SSE 경로: ``api.routes_chat.chunk_to_sse_entries``
- UI 직접 경로: ``ui.components.streaming.consume_stream_into_message``

공유 매핑(``chunk_to_stream_events``)이 어느 한쪽 소비자에서 drifting 하면
이 파일이 계약 위반을 조기 감지한다. 병렬로 착지한 세 모듈(
``stream_events.py``/``routes_chat.py``/``streaming.py``)의 구현이 전제다.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import src.ui.components.streaming as streaming_mod

from api.routes_chat import chunk_to_sse_entries
from api.stream_events import STREAM_EVENT_TYPES, chunk_to_stream_events
from api.streaming_handler import StreamChunk


class _FakeSessionManager(MagicMock):
    """``consume_stream_into_message``용 SessionManager 스텁.

    ``MagicMock`` 상속으로 미구현 정적 메서드 호출을 안전하게 허용하면서,
    ``get``/``set``/``add_message``/``get_messages``/``reset`` 은 실제 은닉
    저장소 위에서 동작시킨다. ``add_message`` 는 실제 구현과 동일하게
    ``msg_id`` 기준 업서트를 수행해 플레이스홀더 메시지를 최종 메시지로
    갱신한다 (``_target_message_dict`` 가 최종 상태를 읽을 수 있게).
    """

    _messages: list[dict[str, Any]] = []
    _store: dict[str, Any] = {}

    @classmethod
    def reset(cls) -> None:
        cls._messages = []
        cls._store = {}

    @classmethod
    def get(
        cls,
        key: str,
        default: Any = None,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        return cls._store.get(key, default)

    @classmethod
    def set(cls, key: str, value: Any, **kwargs: Any) -> None:
        cls._store[key] = value

    @classmethod
    def add_message(
        cls,
        role: str,
        content: str,
        msg_type: str = "general",
        session_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        message: dict[str, Any] = {
            "msg_id": kwargs.pop("msg_id", str(uuid.uuid4())),
            "role": role,
            "content": content,
            "msg_type": msg_type,
            **kwargs,
        }
        for index, existing in enumerate(cls._messages):
            if existing.get("msg_id") == message["msg_id"]:
                cls._messages[index].update(message)
                return
        cls._messages.append(message)

    @classmethod
    def get_messages(cls, session_id: str | None = None) -> list[dict[str, Any]]:
        return list(cls._messages)


def _make_doc(page: int = 1, content: str = "doc1") -> object:
    """``_doc_to_source`` 가 요구하는 Document-like 객체를 생성한다."""
    from langchain_core.documents import Document

    return Document(page_content=content, metadata={"page": page})


def _fully_populated_chunk() -> StreamChunk:
    """여섯 가지 이벤트를 모두 생성하는 super-chunk."""
    return StreamChunk(
        content="안녕하세요",
        status=".retrieve",
        node_name="retriever",
        thought="검색 결과를 분석합니다",
        metadata={"documents": [_make_doc()]},
        citations=[{"page": 1, "text": "인용"}],
        performance={"tokens_per_second": 42.0},
    )


def test_chunk_to_stream_events_canonical_order() -> None:
    """완전 청크 → 이벤트 6개, 타입이 STREAM_EVENT_TYPES 순서 그대로."""
    events = chunk_to_stream_events(_fully_populated_chunk())

    assert len(events) == 6
    assert [ev.type for ev in events] == list(STREAM_EVENT_TYPES)


def test_chunk_to_stream_events_empty_chunk() -> None:
    """빈 StreamChunk → 이벤트 목록이 비어 있어야 한다."""
    assert chunk_to_stream_events(StreamChunk()) == []


def test_chunk_to_sse_entries_emits_all_six_types() -> None:
    """SSE 경로: 엔트리 6개, 타입 순서·이벤트 id(0~5)·counter(6) 일치."""
    entries, counter = chunk_to_sse_entries(_fully_populated_chunk(), 0)

    assert len(entries) == 6
    assert [t for t, _payload, _idx in entries] == list(STREAM_EVENT_TYPES)
    assert [idx for _t, _payload, idx in entries] == list(range(6))
    assert counter == 6


def test_chunk_to_sse_entries_sources_truncates_docs() -> None:
    """sources 페이로드만 ``_doc_to_source(d, max_chars=100)`` 로 변환된다."""
    stub_return: dict[str, Any] = {"page": 1, "content": "x"}
    chunk = StreamChunk(metadata={"documents": [MagicMock()]})

    with patch(
        "api.routes_chat._doc_to_source", return_value=stub_return
    ) as mock_doc_to_source:
        entries, _counter = chunk_to_sse_entries(chunk, 0)

    mock_doc_to_source.assert_called_once()
    assert mock_doc_to_source.call_args.kwargs.get("max_chars") == 100
    sources_entry = next(entry for entry in entries if entry[0] == "sources")
    assert sources_entry[1] == {"documents": [stub_return]}


def test_ui_consumer_honors_all_six_event_kinds() -> None:
    """UI 경로: 여섯 종류의 청크를 소비해 메시지에 전부 반영한다."""
    _FakeSessionManager.reset()
    fake_doc = _make_doc(page=7, content="문서 본문")
    status_text = "status_node"
    thought_text = "심층 추론 과정"
    accumulated_text = "누적된 응답 본문"
    citation: dict[str, Any] = {"source": "S"}
    metrics: dict[str, Any] = {"token_count": 5}

    chunks = [
        StreamChunk(status=status_text, node_name="retriever"),
        StreamChunk(content=accumulated_text),
        StreamChunk(thought=thought_text),
        StreamChunk(metadata={"documents": [fake_doc]}),
        StreamChunk(citations=[citation]),
        StreamChunk(performance=metrics),
    ]

    with (
        patch.object(streaming_mod, "stream_chunks", return_value=iter(chunks)),
        patch.object(streaming_mod, "SessionManager", _FakeSessionManager),
    ):
        result = streaming_mod.consume_stream_into_message(
            "test_sid", "테스트 질문", "test-model", on_chunk=None
        )

    assert result is not None
    assert status_text in result["process_steps"]
    assert result["thought"] == thought_text
    assert result["documents"] == [fake_doc]
    assert result["citations"] == [citation]
    assert result["metrics"] == metrics
    assert result["content"] == accumulated_text


def test_event_set_identical_across_paths() -> None:
    """두 경로가 동일한 이벤트 집합(STREAM_EVENT_TYPES)을 생성한다."""
    chunk = _fully_populated_chunk()

    ui_types = {ev.type for ev in chunk_to_stream_events(chunk)}
    sse_entries, _counter = chunk_to_sse_entries(chunk, 0)
    sse_types = {t for t, _payload, _idx in sse_entries}

    assert ui_types == set(STREAM_EVENT_TYPES)
    assert sse_types == set(STREAM_EVENT_TYPES)
    assert ui_types == sse_types
