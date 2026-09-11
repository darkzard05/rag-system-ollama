"""T14: 스트리밍 상태 채널(on_status) + 캡션 라이프사이클 단위 테스트.

UX-1/UX-2 수정(T10/T11)이 ``_content_generator`` 에 추가한 ``on_status``
콜백 계약을 검증한다:

- 상태 청크(``status`` 존재 + ``content=""``)는 yield되지 않고 on_status 로만
  보고된다 (콘텐츠 누출 없음).
- 연속으로 동일한 상태 텍스트는 1회만 보고된다 (dedupe).
- 첫 콘텐츠 청크에서 "응답 생성 중..." 이 1회 보고된다.
- ``on_status`` 미지정(기존 4-인자 호출) 시 yield 계약은 불변이다.
- ``_render_streaming_with_write_stream`` 의 상태 캡션은 생성 직후 표시되고
  write_stream 종료(정상/예외/■ 중지) 시 ``status_ph.empty()`` 로 정리되며,
  영속 어시스턴트 메시지는 결합된 답변 텍스트를 담는다.

``test_streaming_raw_json_bridge.py`` 의 ``_FakeSessionManager`` 패턴을 로컬
최소 구현으로 미러링한다. src/ 는 수정하지 않는다.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

os.environ.setdefault("IS_CI_TEST", "true")

import ui.components.chat as chat_mod  # noqa: E402
import ui.components.streaming as streaming_mod  # noqa: E402
from api.streaming_handler import StreamChunk  # noqa: E402

_STATUS_A = StreamChunk(content="", status="관련 지식 검색 중...")
_STATUS_B = StreamChunk(content="", status="답변 설계 및 생성 중...")
_CONTENT_1 = StreamChunk(content="안녕")
_CONTENT_2 = StreamChunk(content="하세요")

_ANSWER_TEXT = "안녕하세요"
_INITIAL_CAPTION = "AI가 답변을 생성 중입니다... ▍"


class _FakeSessionManager(MagicMock):
    """``SessionManager`` 의 최소 인메모리 레플리카 (테스트 전용).

    클래스 레벨 ``_store`` 위에서 ``get``/``set``/``add_message``/
    ``get_messages`` 를 동작시키고 ``init_session`` 은 no-op 이다.
    ``MagicMock`` 상속으로 나머지 정적 메서드 호출을 안전하게 허용한다.
    """

    _store: dict[str, Any] = {}

    @classmethod
    def reset(cls) -> None:
        cls._store = {}

    @classmethod
    def init_session(cls, session_id: str | None = None, **kwargs: Any) -> None:
        return None

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
    def set(
        cls, key: str, value: Any, session_id: str | None = None, **kwargs: Any
    ) -> None:
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
        current = cls._store.setdefault("messages", [])
        for index, existing in enumerate(current):
            if existing.get("msg_id") == message["msg_id"]:
                current[index].update(message)
                break
        else:
            current.append(message)

    @classmethod
    def get_messages(cls, session_id: str | None = None) -> list[dict[str, Any]]:
        return list(cls._store.get("messages", []))


def _run_generator(
    chunks: list[StreamChunk],
    *,
    with_on_status: bool = True,
    sid: str = "s",
) -> tuple[list[str], list[str], list[float]]:
    """``_content_generator`` 를 소진하고 (yield, 상태 텍스트, elapsed)를 반환."""
    _FakeSessionManager.reset()
    texts: list[str] = []
    elapsed: list[float] = []

    def _record(text: str, seconds: float) -> None:
        texts.append(text)
        elapsed.append(seconds)

    with (
        patch.object(streaming_mod, "stream_chunks", return_value=iter(chunks)),
        patch.object(streaming_mod, "SessionManager", _FakeSessionManager),
    ):
        if with_on_status:
            out = list(
                streaming_mod._content_generator(
                    "q", "m", sid, "mid", on_status=_record
                )
            )
        else:
            out = list(streaming_mod._content_generator("q", "m", sid, "mid"))
    return out, texts, elapsed


def test_content_generator_reports_statuses_in_order() -> None:
    """상태 → 콘텐츠 → 상태 → 콘텐츠: yield는 콘텐츠만, 상태는 순서대로 보고."""
    out, texts, elapsed = _run_generator([_STATUS_A, _CONTENT_1, _STATUS_B, _CONTENT_2])

    assert out == ["안녕", "하세요"]
    assert texts == [_STATUS_A.status, "응답 생성 중...", _STATUS_B.status]
    assert all(isinstance(value, float) for value in elapsed)
    assert elapsed == sorted(elapsed)


def test_content_generator_dedupes_consecutive_identical_statuses() -> None:
    """동일 상태가 연속으로 들어오면 1회만 보고된다."""
    _out, texts, _elapsed = _run_generator([_STATUS_A, _STATUS_A, _STATUS_B])

    assert texts == [_STATUS_A.status, _STATUS_B.status]


def test_content_generator_first_content_switches_to_answer_in_progress() -> None:
    """첫 콘텐츠 청크에서 "응답 생성 중..." 이 1회 보고된다."""
    out, texts, _elapsed = _run_generator([_CONTENT_1])

    assert texts == ["응답 생성 중..."]
    assert out == ["안녕"]


def test_content_generator_status_events_do_not_leak_into_content() -> None:
    """상태 전용 청크는 yield되지 않고, 각 상태가 on_status 로 보고된다."""
    out, texts, _elapsed = _run_generator([_STATUS_A, _STATUS_B])

    assert out == []
    assert texts == [_STATUS_A.status, _STATUS_B.status]


def test_content_generator_without_on_status_keeps_existing_contract() -> None:
    """on_status 없이(4-인자) 호출해도 yield 계약은 불변이다."""
    out, texts, _elapsed = _run_generator(
        [_STATUS_A, _CONTENT_1, _STATUS_B, _CONTENT_2], with_on_status=False
    )

    assert out == ["안녕", "하세요"]
    assert "".join(out) == _ANSWER_TEXT
    assert texts == []


def _join_stream(gen: Iterator[str]) -> str:
    """실제 ``st.write_stream`` 계약: 제너레이터를 소진해 전체 텍스트 반환."""
    return "".join(gen)


def _make_fake_st() -> MagicMock:
    """write_stream 이 제너레이터를 소진하는 최소 Streamlit 하네스."""
    fake_st = MagicMock()
    fake_st.write_stream.side_effect = _join_stream
    return fake_st


def test_status_caption_created_and_cleared_around_write_stream() -> None:
    """상태 캡션은 생성 직후 표시 → on_status로 갱신 → finally에서 정리된다.

    또한 영속화된 어시스턴트 메시지 내용이 결합된 답변 텍스트와 일치함을
    검증한다 (영속 계약 불변).
    """
    _FakeSessionManager.reset()
    fake = _FakeSessionManager()
    fake_st = _make_fake_st()
    chunks = [_STATUS_A, _CONTENT_1, _STATUS_B, _CONTENT_2]

    with (
        patch.object(streaming_mod, "stream_chunks", return_value=iter(chunks)),
        patch.object(streaming_mod, "SessionManager", fake),
        patch.object(chat_mod, "st", fake_st),
        patch.object(chat_mod, "render_generation_expander", MagicMock()),
        patch.object(chat_mod, "SessionManager", fake),
    ):
        chat_mod._render_streaming_with_write_stream({"msg_id": "mid"}, "s", "질문")

    status_ph = fake_st.empty.return_value
    captions = [call.args[0] for call in status_ph.caption.call_args_list if call.args]
    assert captions[0] == _INITIAL_CAPTION
    assert any("응답 생성 중..." in text for text in captions)
    # finally 블록의 정리가 마지막 메서드 호출이어야 한다.
    assert status_ph.method_calls[-1][0] == "empty"

    messages = fake.get_messages()
    assert len(messages) == 1
    assert messages[0]["role"] == "assistant"
    assert messages[0]["content"] == _ANSWER_TEXT
    assert messages[0]["msg_id"] == "mid"
