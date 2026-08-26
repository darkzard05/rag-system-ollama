"""INT-2 (uiux-fix-p1): 스트리밍 중단 시 확정 피드백 검증.

참고: 리팩터(커밋 9514334) 이후 사용자 중단은 별도 콜백 함수 없이
``SessionManager.set("generation_cancel", True)`` 플래그 세팅 +
소비 스레드(streaming.py)의 읽기/확정 경로로 통합되었다. 따라서 이 테스트는
플래그 기반 취소의 (b) 소비 측 확정 순서와 (c) 렌더 상태만 검증한다.

- (b) 소비 스레드 확정 시 "중단됨"(``cancelled=True``) 저장이
       ``generation_cancel`` 클리어보다 먼저 수행되어 최종 상태에 중단 정보와
       누적 부분 콘텐츠가 남는다 (Metis G4 순서 함정 회귀 방지).
- (c) 렌더 조건: 스트리밍 중 ``generation_cancel=True``면 영속 익스팬더의
       스피너에 "Stopping..."이 표시되고, 확정된(cancelled) 메시지는
       "Stopped · Partial answer preserved" 캡션이 표시된다.
"""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from core.session import SessionManager


def _fake_chunk(
    content: str = "",
    thought: str = "",
    status: str | None = None,
    metadata: dict | None = None,
    performance: dict | None = None,
) -> SimpleNamespace:
    """StreamChunk와 동일한 속성을 가진 경량 fake 청크를 생성합니다."""
    return SimpleNamespace(
        status=status,
        thought=thought,
        content=content,
        metadata=metadata,
        performance=performance,
    )


def test_cancel_flow_saves_cancelled_field_and_preserves_partial_content():
    """(b) 중단 확정: cancelled 저장이 generation_cancel 클리어보다 먼저 수행된다.

    중단 요청 시점을 스트림 중간으로 결정적으로 잡아, 소비 스레드가 플래그를
    감지하고 finally에서 확정 상태(cancelled=True + 누적 부분 콘텐츠)를 저장한 뒤
    플래그를 클리어하는지 검증한다. 클리어를 먼저 수행하는 코드라면 cancelled
    필드가 소실되어 이 테스트가 실패한다 (G4 순서 함정).
    """
    sid = "test_cancel_flow"
    SessionManager.reset_all_state(sid)
    SessionManager.set_session_id(sid)
    SessionManager.set("is_generating_answer", True, sid)

    first_chunk_consumed = threading.Event()
    release_stream = threading.Event()

    # 첫 청크를 내고 멈춘 뒤, 취소 플래그가 세팅되면 후속 청크 없이 종료한다.
    # consume_stream_into_message는 첫 청크를 accumulate한 뒤 제너레이터 종료를
    # 맞이하고, 성공 분기에서 generation_cancel=True를 cancelled로 확정한다.
    def controlled_stream(query: str, model_name: str, session_id: str):
        yield _fake_chunk(status="생성 중...", content="부분 ")
        first_chunk_consumed.set()
        release_stream.wait(timeout=5.0)
        if SessionManager.get("generation_cancel", False, session_id=session_id):
            return  # 취소 확정 → 후속 청크 미제공
        yield _fake_chunk(content="답변")

    def request_cancel_after_first_chunk() -> None:
        first_chunk_consumed.wait(timeout=5.0)
        SessionManager.set("generation_cancel", True, session_id=sid)
        release_stream.set()

    from ui.components.streaming import consume_stream_into_message

    with patch("ui.components.streaming.stream_chunks", controlled_stream):
        canceller = threading.Thread(target=request_cancel_after_first_chunk)
        canceller.start()
        consume_stream_into_message(sid, "질문", "test-model")
        canceller.join(timeout=5.0)

    # 확정 상태 저장이 클리어보다 먼저 수행되었으므로 최종 상태에 중단 정보가 남는다
    assert SessionManager.get("generation_cancel", False, sid) is False
    assert SessionManager.get("is_generating_answer", False, sid) is False

    messages = SessionManager.get_messages(session_id=sid)
    target = next(m for m in messages if m.get("role") == "assistant")
    assert target["cancelled"] is True
    assert target["msg_type"] == "general"
    assert target["content"] == "부분 "  # 누적 부분 콘텐츠 보존 (후속 청크 미반영)


def test_streaming_status_shows_cancelling_caption():
    """(c) 스트리밍 중 generation_cancel=True → 영속 익스팬더 라벨/스피너에 "중단 중..." 표시."""
    from ui.components import chat as chat_mod

    sid = "test_cancel_render"
    SessionManager.reset_all_state(sid)
    SessionManager.set_session_id(sid)
    SessionManager.add_message(
        role="assistant",
        content="부분 답변",
        msg_type="streaming",
        msg_id="m1",
        thought="",
        documents=[],
        metrics={},
        processed_content=None,
        session_id=sid,
    )
    SessionManager.set("generation_cancel", True, sid)

    with patch("ui.components.chat.st") as mock_st:
        mock_st.chat_message.return_value.__enter__.return_value = MagicMock()
        expander_holder = MagicMock()
        mock_st.expander.return_value.__enter__.return_value = expander_holder
        spinner_holder = MagicMock()
        mock_st.spinner.return_value.__enter__.return_value = spinner_holder
        mock_st.button.return_value = None
        # 폴링 fragment가 제거된 표준 구조에서는 _render_unified_timeline을
        # 직접 호출해 렌더 경로를 검증한다.
        chat_mod._render_unified_timeline(sid)

    # 영속 익스팬더 라벨은 "Answer details"(상수)이고, "Stopping..."은
    # 스트리밍 중 스피너 텍스트로만 노출된다.
    expander_labels = [c.args[0] for c in mock_st.expander.call_args_list]
    assert expander_labels
    assert "Answer details" in expander_labels[0]
    spinner_text = mock_st.spinner.call_args.args[0]
    assert "Stopping..." in spinner_text


def test_render_message_shows_cancelled_caption():
    """(c) 확정(cancelled) 메시지: "중단됨" 캡션 표시, 완료 문구 미표시."""
    from ui.components import chat as chat_mod

    with patch("ui.components.chat.st") as mock_st:
        mock_st.chat_message.return_value.__enter__.return_value = MagicMock()
        mock_st.expander.return_value.__enter__.return_value = MagicMock()
        mock_st.button.return_value = False

        chat_mod.render_message(
            role="assistant",
            content="부분 답변",
            msg_id="x",
            msg_type="general",
            cancelled=True,
        )

    captions = [c.args[0] for c in mock_st.caption.call_args_list]
    assert any("Stopped · Partial answer preserved" in c for c in captions)
    assert not any("Answer complete" in c for c in captions)
