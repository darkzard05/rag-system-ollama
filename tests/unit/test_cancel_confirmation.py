"""INT-2 (uiux-fix-p1): 스트리밍 중단 시 확정 피드백 검증.

- (a) ``_handle_stop_generation``이 ``generation_cancel=True``를 즉시 세팅한다.
- (b) 소비 스레드 확정 시 "중단됨"(``cancelled=True``) 저장이
      ``generation_cancel`` 클리어보다 먼저 수행되어 최종 상태에 중단 정보와
      누적 부분 콘텐츠가 남는다 (Metis G4 순서 함정 회귀 방지).
- (c) 렌더 조건: 스트리밍 중 ``generation_cancel=True``면 상태 박스에
      "중단 중..."이 표시되고, 확정된(cancelled) 메시지는 "중단됨" 캡션이
      표시된다.
"""

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from core.session import SessionManager
from ui.components.chat import _handle_stop_generation


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
    """is_generating_answer가 False가 될 때까지 폴링합니다 (데드락 감지)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        raw_state = SessionManager._fallback_sessions.get(sid, {})
        if raw_state.get("is_generating_answer", True) is False:
            break
        time.sleep(0.05)
    assert SessionManager.get("is_generating_answer", False, sid) is False


def test_handle_stop_generation_sets_cancel_flag_immediately():
    """(a) 중단 버튼 콜백: generation_cancel=True 즉시 세팅 + rerun."""
    sid = "test_cancel_flag"
    SessionManager.reset_all_state(sid)
    SessionManager.set("generation_cancel", False, sid)

    with patch("ui.components.chat.st.rerun") as mock_rerun:
        _handle_stop_generation(sid)

    assert SessionManager.get("generation_cancel", False, sid) is True
    mock_rerun.assert_called_once()


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

    def controlled_stream(query: str, model_name: str, session_id: str):
        yield _fake_chunk(status="생성 중...", content="부분 ")
        first_chunk_consumed.set()
        release_stream.wait(timeout=5.0)
        yield _fake_chunk(content="답변")

    def request_cancel_after_first_chunk() -> None:
        first_chunk_consumed.wait(timeout=5.0)
        SessionManager.set("generation_cancel", True, session_id=sid)
        release_stream.set()

    from ui.components.streaming import start_streaming_turn

    with patch("ui.components.streaming.stream_chunks", controlled_stream):
        msg_id = start_streaming_turn(sid, "질문", "test-model")
        canceller = threading.Thread(target=request_cancel_after_first_chunk)
        canceller.start()
        _wait_for_flag_cleared(sid)
        canceller.join(timeout=5.0)

    # 확정 상태 저장이 클리어보다 먼저 수행되었으므로 최종 상태에 중단 정보가 남는다
    assert SessionManager.get("generation_cancel", False, sid) is False
    assert SessionManager.get("is_generating_answer", False, sid) is False

    messages = SessionManager.get_messages(session_id=sid)
    target = next(m for m in messages if m.get("msg_id") == msg_id)
    assert target["cancelled"] is True
    assert target["msg_type"] == "general"
    assert target["content"] == "부분 "  # 누적 부분 콘텐츠 보존 (후속 청크 미반영)


def test_streaming_status_shows_cancelling_caption():
    """(c) 스트리밍 중 generation_cancel=True → 상태 박스에 "중단 중..." 표시."""
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
        status_holder = MagicMock()
        mock_st.status.return_value.__enter__.return_value = status_holder
        mock_st.button.return_value = None
        # fragment 래퍼는 실행 컨텍스트가 없으면 본문을 실행하지 않으므로
        # 원본 함수(__wrapped__)를 직접 호출한다.
        chat_mod._render_unified_timeline.__wrapped__(sid)

    label = mock_st.status.call_args.args[0]
    assert "중단 중..." in label
    captions = [c.args[0] for c in status_holder.caption.call_args_list]
    assert "중단 중..." in captions


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
    assert any("중단됨" in c for c in captions)
    assert not any("✅ 답변 생성 완료" in c for c in captions)
