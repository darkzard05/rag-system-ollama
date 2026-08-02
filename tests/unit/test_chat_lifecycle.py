"""채팅 턴 라이프사이클 검증: 스트리밍 완료 후 영속화(persist_completed_turn) 동작.

Phase A에서 실제 구조(render_streaming_area + persist_completed_turn)에 맞게
영속화 단계의 상태 전이를 검증하는 테스트로 복구함.
Phase B: StreamingResult.cancelled 필드 제거 → result dict에서 해당 키 제거,
documents 포함 턴(pdf_annotations/pdf_target_page 부작용) 케이스 추가.
"""

from unittest.mock import MagicMock, patch

from core.session import SessionManager
from ui.components.streaming import StreamingResult, persist_completed_turn


def test_persist_completed_turn_stores_assistant_message():
    """정상 완료 턴: 어시스턴트 메시지 저장 + is_generating_answer 리셋."""
    session_id = "test_session"
    SessionManager.reset_all_state(session_id)
    SessionManager.set("is_generating_answer", True, session_id)

    result: StreamingResult = {
        "content": "Hello world",
        "thought": "I am thinking",
        "documents": [],
        "performance": {"total_time": 1.0},
        "processed_content": "Hello world",
        "error": None,
    }

    with patch("ui.components.streaming.st") as mock_st:
        mock_st.rerun = MagicMock()
        persist_completed_turn(session_id, result)

    messages = SessionManager.get_messages(session_id=session_id)
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == "Hello world"
    assert messages[-1]["thought"] == "I am thinking"
    assert messages[-1]["processed_content"] == "Hello world"
    assert SessionManager.get("is_generating_answer", session_id=session_id) is False
    mock_st.rerun.assert_called_once()


def test_persist_completed_turn_stores_error_message():
    """오류 턴: 오류 메시지 저장 + is_generating_answer 리셋."""
    session_id = "test_session"
    SessionManager.reset_all_state(session_id)
    SessionManager.set("is_generating_answer", True, session_id)

    result: StreamingResult = {
        "content": "",
        "thought": "",
        "documents": [],
        "performance": {},
        "processed_content": None,
        "error": "stream failed",
    }

    with patch("ui.components.streaming.st") as mock_st:
        mock_st.rerun = MagicMock()
        persist_completed_turn(session_id, result)

    messages = SessionManager.get_messages(session_id=session_id)
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == "stream failed"
    assert SessionManager.get("is_generating_answer", session_id=session_id) is False
    mock_st.rerun.assert_called_once()


def test_persist_completed_turn_sets_pdf_side_effects_from_documents():
    """documents 포함 턴: pdf_annotations 저장 + pdf_target_page/current_page 점프."""
    session_id = "test_session_docs"
    SessionManager.reset_all_state(session_id)
    SessionManager.set("is_generating_answer", True, session_id)

    doc = MagicMock()
    doc.metadata = {"page": 7}

    result: StreamingResult = {
        "content": "정리된 답변입니다.",
        "thought": "",
        "documents": [doc],
        "performance": {"total_time": 1.0},
        "processed_content": "정리된 답변입니다.",
        "error": None,
    }

    with patch("ui.components.streaming.st") as mock_st:
        mock_st.rerun = MagicMock()
        persist_completed_turn(session_id, result)

    messages = SessionManager.get_messages(session_id=session_id)
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["documents"] == [doc]
    assert SessionManager.get("pdf_annotations", session_id=session_id) == []
    assert SessionManager.get("pdf_target_page", session_id=session_id) == 7
    assert SessionManager.get("current_page", session_id=session_id) == 7
    assert SessionManager.get("is_generating_answer", session_id=session_id) is False
    mock_st.rerun.assert_called_once()
