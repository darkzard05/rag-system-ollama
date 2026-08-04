"""채팅 턴 라이프사이클 검증: 스트리밍 소비 완료 후 메시지 영속화·PDF 부작용 동작.

P1에서 persist_completed_turn이 제거되고 동일 로직이
_spawn_stream_consumer의 finally 블록에 인라인되었으므로, start_streaming_turn을
통해 소비 스레드의 실제 동작으로 상태 전이를 검증한다.
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from core.session import SessionManager
from ui.components.streaming import start_streaming_turn


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


def _wait_for_pdf_side_effects(sid: str, timeout: float = 5.0) -> None:
    """PDF 부작용은 플래그 해제 이후 스레드에서 적용되므로 폴링으로 대기합니다."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if SessionManager.get("pdf_annotations", None, session_id=sid) is not None:
            return
        time.sleep(0.02)
    raise AssertionError(f"pdf_annotations가 {timeout}s 내에 설정되지 않음 (sid={sid})")


def test_streaming_completion_stores_assistant_message():
    """정상 완료 턴: 어시스턴트 메시지 저장 + is_generating_answer 리셋."""
    session_id = "test_session"
    SessionManager.reset_all_state(session_id)
    SessionManager.set("is_generating_answer", True, session_id)

    fake_chunks = [
        _fake_chunk(
            status="생성 중...", thought="I am thinking", content="Hello world"
        ),
    ]
    with patch(
        "ui.components.streaming.stream_chunks",
        return_value=iter(fake_chunks),
    ):
        msg_id = start_streaming_turn(session_id, "질문", "test-model")

    _wait_for_flag_cleared(session_id)

    messages = SessionManager.get_messages(session_id=session_id)
    target = next(m for m in messages if m.get("msg_id") == msg_id)
    assert target["role"] == "assistant"
    assert target["content"] == "Hello world"
    assert target["thought"] == "I am thinking"
    assert target["msg_type"] == "general"
    assert SessionManager.get("is_generating_answer", False, session_id) is False


def test_streaming_error_stores_friendly_message_and_clears_flag():
    """오류 턴: 친화적 오류 메시지 저장 + is_generating_answer 리셋."""
    session_id = "test_session_err"
    SessionManager.reset_all_state(session_id)
    SessionManager.set("is_generating_answer", True, session_id)

    def _raising_stream(query: str, model_name: str, session_id: str):
        yield _fake_chunk(status="시작 중", content="부분 응답")
        raise RuntimeError("boom")

    with patch("ui.components.streaming.stream_chunks", _raising_stream):
        msg_id = start_streaming_turn(session_id, "질문", "test-model")

    _wait_for_flag_cleared(session_id)

    messages = SessionManager.get_messages(session_id=session_id)
    target = next(m for m in messages if m.get("msg_id") == msg_id)
    assert target["msg_type"] == "general"
    assert "boom" not in target["error"]
    assert "오류가 발생" in target["error"]
    assert SessionManager.get("is_generating_answer", False, session_id) is False


def test_streaming_documents_set_pdf_side_effects():
    """documents 포함 턴: pdf_annotations 저장 + pdf_target_page/current_page 점프."""
    session_id = "test_session_docs"
    SessionManager.reset_all_state(session_id)
    SessionManager.set("is_generating_answer", True, session_id)
    SessionManager.set("pdf_file_path", "dummy.pdf", session_id)

    doc = MagicMock()
    doc.metadata = {"page": 7}

    fake_chunks = [
        _fake_chunk(content="정리된 답변입니다.", metadata={"documents": [doc]}),
    ]
    with patch(
        "ui.components.streaming.stream_chunks",
        return_value=iter(fake_chunks),
    ):
        msg_id = start_streaming_turn(session_id, "질문", "test-model")

    _wait_for_flag_cleared(session_id)

    messages = SessionManager.get_messages(session_id=session_id)
    target = next(m for m in messages if m.get("msg_id") == msg_id)
    assert target["role"] == "assistant"
    assert target["documents"] == [doc]
    _wait_for_pdf_side_effects(session_id)
    pdf_annotations = SessionManager.get("pdf_annotations", session_id=session_id)
    assert pdf_annotations["file_hash"] is None  # 테스트에서 file_hash 미설정
    assert pdf_annotations["annotations"] == []  # 좌표 없음 → 주석 없음

    pdf_target = SessionManager.get("pdf_target_page", session_id=session_id)
    assert isinstance(pdf_target, dict)
    assert pdf_target["page"] == 7
    assert pdf_target["source"] == "auto"
    assert isinstance(pdf_target["ts"], float)

    assert SessionManager.get("current_page", session_id=session_id) == 7
    assert SessionManager.get("is_generating_answer", False, session_id) is False
