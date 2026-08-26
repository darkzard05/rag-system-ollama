"""채팅 턴 라이프사이클 검증: 스트리밍 소비 완료 후 메시지 영속화·PDF 부작용 동작.

표준 리팩터 이후 스트리밍은 백그라운드 스레드가 아닌 ``consume_stream_into_message``
순수 헬퍼를 통한 동기 소비로 처리된다. 헬퍼가 실제 상태 전이(영속화·플래그 해제)를
수행하므로 이를 직접 호출해 검증한다.
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from core.session import SessionManager
from ui.components.streaming import consume_stream_into_message


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
        msg = consume_stream_into_message(session_id, "질문", "test-model")

    messages = SessionManager.get_messages(session_id=session_id)
    target = next(m for m in messages if m.get("msg_id") == msg["msg_id"])
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
        msg = consume_stream_into_message(session_id, "질문", "test-model")

    messages = SessionManager.get_messages(session_id=session_id)
    target = next(m for m in messages if m.get("msg_id") == msg["msg_id"])
    assert target["msg_type"] == "general"
    assert "boom" not in target["error"]
    assert "An error occurred" in target["error"]
    assert SessionManager.get("is_generating_answer", False, session_id) is False


def test_streaming_documents_set_pdf_side_effects():
    """documents 포함 턴: pdf_annotations 저장 + 자동 점프 미발생 (uiux-fix-p1 INT-1)."""
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
        msg = consume_stream_into_message(session_id, "질문", "test-model")

    messages = SessionManager.get_messages(session_id=session_id)
    target = next(m for m in messages if m.get("msg_id") == msg["msg_id"])
    assert target["role"] == "assistant"
    assert target["documents"] == [doc]
    _wait_for_pdf_side_effects(session_id)
    pdf_annotations = SessionManager.get("pdf_annotations", session_id=session_id)
    assert pdf_annotations["file_hash"] is None  # 테스트에서 file_hash 미설정
    assert pdf_annotations["annotations"] == []  # 좌표 없음 → 주석 없음

    # uiux-fix-p1 INT-1: 답변 완료 시 자동 점프 토큰/페이지가 세팅되지 않는다.
    # (page=7 문서가 있어도 사용자 발의 없는 화면 이동 금지)
    assert SessionManager.get("pdf_target_page", session_id=session_id) is None
    assert SessionManager.get("current_page", session_id=session_id) == 1  # 기본값 유지
    assert SessionManager.get("is_generating_answer", False, session_id) is False
