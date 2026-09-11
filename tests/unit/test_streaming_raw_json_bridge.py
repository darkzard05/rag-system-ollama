"""raw JSON 누출 수정(T9) 브릿지 테스트.

structured 모드(``raw_json`` 청크)에서 ``final_answer`` 값만 흘려보내는 세
소비 경로 — ``_FinalAnswerExtractor``(헬퍼), ``_content_generator``,
``stream_content`` — 와 ``_render_streaming_with_write_stream`` 영속 체인을
검증한다. 계획 ``.omo/plans/raw-json-leak-fix-plan.md`` §6 T9 스펙 그대로:

- 3분할 샘플(c1/c2/c3)이 키 경계(``"final_answer"``)와 값 경계(이스케이프
  개행)를 모두 가로지른다.
- 모든 산출물에서 raw JSON 스캐폴드(``{``/``}``/``:``/``"reasoning"``/
  ``"final_answer"``)가 나타나지 않아야 한다.
- ``generation_cancel`` 시점에는 부분 delta만 yield되고 ``finally`` 가
  aux 상태를 완료(``complete: True``)로 기록한다.

``chat_mod`` 는 ``ui.components.streaming`` 의 ``_content_generator`` 를
사용하므로, ``streaming_mod`` 도 동일 인스턴스(``ui.components.streaming``)
를 가리켜 패치가 실제 해석 경로에 적용되게 한다. src/는 수정하지 않는다.
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

_FINAL_ANSWER = "안녕하세요, 세상!"
_FINAL_ANSWER_NEWLINE = "안녕하세요, 세상!\n"

# 샘플 raw JSON 3분할 (Plan §6 T9): 키 경계 c1→c2, 값 경계 c2→c3.
_C1 = '{"reasoning":"검토 중","final_ans'
_C2 = 'wer":"안녕하세요, 세상!'
_C3 = '\\n","citations":[],"confidence":0.9}'

_RAW_JSON_CHUNKS = [
    StreamChunk(content=_C1, raw_json=True),
    StreamChunk(content=_C2, raw_json=True),
    StreamChunk(content=_C3, raw_json=True),
]

_PLAIN_CHUNK = StreamChunk(content=" (추가)")

_JSON_SCAFFOLD_TOKENS = ("{", "}", ":", '"reasoning"', '"final_answer"')


class _FakeSessionManager(MagicMock):
    """``SessionManager``의 최소 인메모리 레플리카 (테스트 전용).

    ``MagicMock`` 상속으로 미구현 정적 메서드 호출을 안전하게 허용하면서,
    ``get``/``set``/``add_message``/``get_messages``/``reset`` 은 실제 은닉
    저장소(클래스 레벨 ``_store``) 위에서 동작시킨다. ``add_message`` 는 실제
    구현과 동일하게 ``msg_id`` 기준 업서트를 수행하며 영속 메시지는
    ``_store["messages"]`` 에 순서대로 쌓인다. 실제와 같은 세션 단위
    락/분리는 없고, ``get`` 은 3번째 위치 인자(``session_id``)를 그대로 받는다
    (``get("generation_cancel", False, sid)`` 호출 형태 대응).
    """

    _store: dict[str, Any] = {}
    _messages: list[dict[str, Any]] = []

    @classmethod
    def reset(cls) -> None:
        cls._store = {}
        cls._messages = []

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
        cls._messages = current

    @classmethod
    def get_messages(cls, session_id: str | None = None) -> list[dict[str, Any]]:
        return list(cls._messages)


def _assert_no_json_scaffold(text: str) -> None:
    """raw JSON 스캐폴드 토큰이 하나도 없는지 검증한다."""
    for token in _JSON_SCAFFOLD_TOKENS:
        assert token not in text


def _join_stream(gen: Iterator[str]) -> str:
    """실제 ``st.write_stream`` 계약: 제너레이터를 소진해 전체 텍스트 반환."""
    return "".join(gen)


def _make_fake_st() -> MagicMock:
    fake_st = MagicMock()
    fake_st.write_stream.side_effect = _join_stream
    fake_st.chat_message = MagicMock()
    return fake_st


def _render_with_write_stream(fake: _FakeSessionManager, fake_st: MagicMock) -> None:
    """``_render_streaming_with_write_stream`` 5종 패치 하네스 (test 6과 동일)."""
    with (
        patch.object(
            streaming_mod, "stream_chunks", return_value=iter(_RAW_JSON_CHUNKS)
        ),
        patch.object(streaming_mod, "SessionManager", fake),
        patch.object(chat_mod, "st", fake_st),
        patch.object(chat_mod, "render_generation_expander", MagicMock()),
        patch.object(chat_mod, "SessionManager", fake),
    ):
        chat_mod._render_streaming_with_write_stream({"msg_id": "mid"}, "s", "질문")


def test_final_answer_extractor_incremental_join_equals_final_answer() -> None:
    """``_FinalAnswerExtractor.feed`` c1/c2/c3 → 정확한 delta 시퀀스 + join.

    c1(키 미도착)은 ``""``, c2는 ``"안녕하세요, 세상!"``, c3는 ``"\\n"``
    (이스케이프 개행 디코드). 비어 있지 않은 delta만 모으면 계획의 시퀀스와
    일치하고 join 은 final_answer 값과 같다. 어떤 delta에도 raw JSON
    스캐폴드가 없어야 한다 (D-A/D-D 불변식).
    """
    extractor = streaming_mod._FinalAnswerExtractor()

    deltas = [extractor.feed(part) for part in (_C1, _C2, _C3)]

    assert deltas == ["", _FINAL_ANSWER, "\n"]
    non_empty = [delta for delta in deltas if delta]
    assert non_empty == [_FINAL_ANSWER, "\n"]
    assert "".join(non_empty) == _FINAL_ANSWER_NEWLINE
    for delta in non_empty:
        _assert_no_json_scaffold(delta)


def test_content_generator_yields_only_final_answer_for_raw_json() -> None:
    """raw_json 청크 3개 → ``_content_generator`` 가 delta 2개만 yield."""
    with (
        patch.object(
            streaming_mod, "stream_chunks", return_value=iter(_RAW_JSON_CHUNKS)
        ),
        patch.object(streaming_mod, "SessionManager", _FakeSessionManager),
    ):
        out = list(streaming_mod._content_generator("q", "m", "s", "mid"))

    assert out == [_FINAL_ANSWER, "\n"]
    assert len(out) == 2
    assert "".join(out) == _FINAL_ANSWER_NEWLINE
    for piece in out:
        _assert_no_json_scaffold(piece)


def test_content_generator_mixed_raw_json_and_plain_keeps_plain() -> None:
    """raw_json 3개 + 일반 청크 → 일반 청크는 원문 그대로 yield."""
    chunks = [*_RAW_JSON_CHUNKS, _PLAIN_CHUNK]
    with (
        patch.object(streaming_mod, "stream_chunks", return_value=iter(chunks)),
        patch.object(streaming_mod, "SessionManager", _FakeSessionManager),
    ):
        out = list(streaming_mod._content_generator("q", "m", "s", "mid"))

    assert out == [_FINAL_ANSWER, "\n", " (추가)"]
    assert "".join(out) == _FINAL_ANSWER_NEWLINE + " (추가)"
    assert out[-1] == " (추가)"


def test_stream_content_yields_only_final_answer_for_raw_json() -> None:
    """content-only 브릿지 ``stream_content`` 도 delta만 yield (no 누적)."""
    with (
        patch.object(
            streaming_mod, "stream_chunks", return_value=iter(_RAW_JSON_CHUNKS)
        ),
        patch.object(streaming_mod, "SessionManager", _FakeSessionManager),
    ):
        out = list(streaming_mod.stream_content("q", "m", "s"))

    assert out == [_FINAL_ANSWER, "\n"]
    assert len(out) == 2
    assert "".join(out) == _FINAL_ANSWER_NEWLINE
    for piece in out:
        _assert_no_json_scaffold(piece)


def test_stream_content_mixed_raw_json_and_plain_keeps_plain() -> None:
    """``stream_content`` 혼합: plain 청크는 원문 그대로 yield."""
    chunks = [*_RAW_JSON_CHUNKS, _PLAIN_CHUNK]
    with (
        patch.object(streaming_mod, "stream_chunks", return_value=iter(chunks)),
        patch.object(streaming_mod, "SessionManager", _FakeSessionManager),
    ):
        out = list(streaming_mod.stream_content("q", "m", "s"))

    assert out == [_FINAL_ANSWER, "\n", " (추가)"]
    assert "".join(out) == _FINAL_ANSWER_NEWLINE + " (추가)"


def test_render_streaming_write_stream_persists_final_answer_not_raw_json() -> None:
    """영속 체인: 어시스턴트 메시지 content 는 final_answer 텍스트만 저장.

    ``st.write_stream`` 이 제너레이터를 소진해 ``content`` 를 반환
    (``_join_stream``) → ``str(response)`` 가 raw JSON이 아닌 순수 delta로
    영속된다. ``msg_id`` 는 ``"mid"`` 유지, write_stream 은 1회 호출.
    """
    fake = _FakeSessionManager()
    fake.reset()

    _render_with_write_stream(fake, _make_fake_st())

    messages = fake.get_messages()
    assert len(messages) == 1
    message = messages[0]
    assert message["role"] == "assistant"
    assert message["content"] == _FINAL_ANSWER_NEWLINE
    assert message["msg_id"] == "mid"
    assert "{" not in message["content"]
    assert '"reasoning"' not in message["content"]
    _assert_no_json_scaffold(message["content"])


def test_content_generator_cancel_yields_partial_final_answer() -> None:
    """취소 의미론: 3번째 루프 검사(→c3 직전)부터 ``generation_cancel=True``.

    Phase A (제너레이터): c3의 ``"\\n"`` 은 처리되지 않아 부분 delta
    ``["안녕하세요, 세상!"]`` 만 yield 되고, ``finally`` 가 aux 를
    ``complete: True`` 로 기록한다.
    Phase B (영속): chat 렌더 경유로 부분 delta가 그대로 영속되며
    ``cancelled`` 플래그가 True 로 저장된다.
    """
    fake = _FakeSessionManager()
    fake.reset()
    calls = {"n": 0}
    orig_get = _FakeSessionManager.get  # 클래스 접근 → cls 바인딩된 bound method

    def flaky_get(
        key: str,
        default: Any = None,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        if key == "generation_cancel":
            calls["n"] += 1
            if calls["n"] >= 3:
                return True
        return orig_get(key, default, session_id=session_id, **kwargs)

    fake.get = flaky_get  # 인스턴스 속성 — 바인딩 없이 호출 인자 그대로 전달

    # Phase A — 제너레이터 직접 소비.
    with (
        patch.object(
            streaming_mod, "stream_chunks", return_value=iter(_RAW_JSON_CHUNKS)
        ),
        patch.object(streaming_mod, "SessionManager", fake),
    ):
        out = list(streaming_mod._content_generator("q", "m", "s", "mid"))

    assert out == [_FINAL_ANSWER]
    for piece in out:
        assert "\n" not in piece
        _assert_no_json_scaffold(piece)
    aux = fake.get("stream_active_aux", {}, "s")
    assert aux["complete"] is True

    # Phase B — chat 렌더 영속 체인.
    fake.reset()
    calls["n"] = 0
    _render_with_write_stream(fake, _make_fake_st())

    messages = fake.get_messages()
    assert len(messages) == 1
    message = messages[0]
    assert message["content"] == _FINAL_ANSWER
    assert message["cancelled"] is True
    assert message["msg_id"] == "mid"
