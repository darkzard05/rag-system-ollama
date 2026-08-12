"""P0 회귀 테스트: 스트리밍 소비 스레드 데드락 + 타임라인 오류 표시.

- Bug 1: `_spawn_stream_consumer`의 finally 블록에서 세션 락(비재진입
  threading.Lock) 보유 중 `SessionManager.set()`을 호출하여 데드락 발생,
  `is_generating_answer`가 True로 고정되는 문제의 회귀 방지.
- Bug 2: 스트리밍 실패 시 `msg["error"]`가 타임라인에 표시되지 않아
  사용자에게 보이지 않던 문제의 회귀 방지.
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

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


def _wait_for_flag_cleared(sid: str, timeout: float = 5.0) -> None:
    """is_generating_answer가 False가 될 때까지 폴링합니다 (데드락 감지).

    폴링은 락 없는 raw 읽기로 수행한다. 데드락이 재발하면 소비 스레드가
    세션 락을 영원히 보유하므로 SessionManager.get()은 락 획득에서 블로킹되어
    테스트가 중단(hang)된다. 락 없는 읽기로 폴링해야 제한 시간 내에
    깔끔하게 실패(fail)할 수 있다.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        raw_state = SessionManager._fallback_sessions.get(sid, {})
        if raw_state.get("is_generating_answer", True) is False:
            break
        time.sleep(0.05)
    else:
        pytest.fail(
            "is_generating_answer가 제한 시간 내에 해제되지 않음 "
            f"({timeout}s) - 세션 락 데드락 가능성"
        )

    # 최종 확인은 공개 API로 수행 (락을 정상적으로 획득할 수 있어야 함)
    assert SessionManager.get("is_generating_answer", False, sid) is False


def _wait_for_pdf_side_effects(sid: str, timeout: float = 5.0) -> None:
    """PDF 부작용은 플래그 해제 이후 스레드에서 적용되므로 폴링으로 대기합니다."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if SessionManager.get("pdf_annotations", None, session_id=sid) is not None:
            return
        time.sleep(0.02)
    pytest.fail(f"pdf_annotations가 {timeout}s 내에 설정되지 않음 (sid={sid})")


def test_consumer_completes_sets_generating_false_no_deadlock():
    """정상 완료: finally 블록이 is_generating_answer를 해제해야 한다.

    락 보유 중 SessionManager.set() 재호출(데드락)이 제거되었는지 검증.
    """
    sid = "test_stream_deadlock_ok"
    SessionManager.reset_all_state(sid)
    SessionManager.set_session_id(sid)
    SessionManager.set("is_generating_answer", True, sid)

    fake_chunks = [
        _fake_chunk(status="생성 중...", thought="생각 중", content="Hello "),
        _fake_chunk(content="world!", performance={"total_time": 1.0}),
    ]

    from ui.components.streaming import start_streaming_turn

    with patch(
        "ui.components.streaming.stream_chunks",
        return_value=iter(fake_chunks),
    ):
        msg_id = start_streaming_turn(sid, "질문", "test-model")

    # 데드락이 없다면 소비 스레드가 플래그를 해제한다
    _wait_for_flag_cleared(sid)
    assert SessionManager.get("is_generating_answer", False, sid) is False

    # 완료된 메시지는 general로 전환되어야 한다
    messages = SessionManager.get_messages(session_id=sid)
    target = next(m for m in messages if m.get("msg_id") == msg_id)
    assert target["msg_type"] == "general"
    assert "Hello world!" in target["content"]


def test_consumer_error_sets_message_error_and_clears_flag():
    """오류 발생: 메시지에 error가 기록되고 is_generating_answer가 해제된다."""
    sid = "test_stream_deadlock_err"
    SessionManager.reset_all_state(sid)
    SessionManager.set_session_id(sid)
    SessionManager.set("is_generating_answer", True, sid)

    def _raising_stream(query: str, model_name: str, session_id: str):
        yield _fake_chunk(status="시작 중", content="부분 응답")
        raise RuntimeError("boom")

    from ui.components.streaming import start_streaming_turn

    with patch("ui.components.streaming.stream_chunks", _raising_stream):
        msg_id = start_streaming_turn(sid, "질문", "test-model")

    _wait_for_flag_cleared(sid)
    assert SessionManager.get("is_generating_answer", False, sid) is False

    messages = SessionManager.get_messages(session_id=sid)
    target = next(m for m in messages if m.get("msg_id") == msg_id)
    assert target["msg_type"] == "general"
    # P0 계약: 원시 예외 문자열은 노출하지 않고 친화적 메시지로 매핑된다
    assert "boom" not in target["error"]
    assert "오류가 발생" in target["error"]


def test_consumer_documents_set_pdf_side_effects():
    """완료 턴: 문서 메타데이터로 pdf_annotations 반영 + 자동 점프 미발생(INT-1)."""
    sid = "test_stream_deadlock_pdf"
    SessionManager.reset_all_state(sid)
    SessionManager.set_session_id(sid)
    SessionManager.set("is_generating_answer", True, sid)
    SessionManager.set("pdf_file_path", "dummy.pdf", sid)

    doc = MagicMock()
    doc.metadata = {"page": 7}
    fake_chunks = [
        _fake_chunk(content="정리된 답변입니다.", metadata={"documents": [doc]}),
    ]

    from ui.components.streaming import start_streaming_turn

    with patch(
        "ui.components.streaming.stream_chunks",
        return_value=iter(fake_chunks),
    ):
        msg_id = start_streaming_turn(sid, "질문", "test-model")

    _wait_for_flag_cleared(sid)
    assert SessionManager.get("is_generating_answer", False, sid) is False

    messages = SessionManager.get_messages(session_id=sid)
    target = next(m for m in messages if m.get("msg_id") == msg_id)
    assert target["documents"] == [doc]
    _wait_for_pdf_side_effects(sid)
    pdf_annotations = SessionManager.get("pdf_annotations", session_id=sid)
    assert pdf_annotations["file_hash"] is None  # 테스트에서 file_hash 미설정
    assert pdf_annotations["annotations"] == []  # 좌표 없음 → 주석 없음

    # uiux-fix-p1 INT-1: 답변 완료 시 자동 점프 토큰/페이지가 세팅되지 않는다.
    # (page=7 문서가 있어도 사용자 발의 없는 화면 이동 금지)
    assert SessionManager.get("pdf_target_page", session_id=sid) is None
    assert SessionManager.get("current_page", session_id=sid) == 1  # 기본값 유지
    assert SessionManager.get("generation_cancel", False, sid) is False


def test_render_message_shows_error_via_st_error():
    """오류 메시지: st.error로 표시되고 본문 markdown은 렌더링되지 않아야 한다."""
    from ui.components import chat as chat_mod

    with patch("ui.components.chat.st") as mock_st:
        # chat_message 컨텍스트 매니저 동작 구성
        mock_st.chat_message.return_value.__enter__.return_value = MagicMock()
        mock_st.expander.return_value.__enter__.return_value = MagicMock()

        chat_mod.render_message(
            role="assistant",
            content="",
            msg_id="x",
            msg_type="general",
            error="boom",
        )

    assert mock_st.error.called
    assert "boom" in str(mock_st.error.call_args.args[0])
    # 오류가 있는 메시지는 본문 콘텐츠를 렌더링하지 않아야 한다
    mock_st.markdown.assert_not_called()
