"""성능 지표 저장·렌더링 검증 (Phase B).

Phase B에서 성능 지표 popover 렌더링은 제거되었습니다. metrics는 세션 메시지에
저장된 채 유지되므로, 여기서는 (1) 렌더링 경로가 popover/성능 테이블 HTML을
더 이상 생성하지 않고, (2) 스트리밍 소비 스레드가 performance 청크를 메시지
metrics로 그대로 저장하는지 검증합니다.
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from core.session import SessionManager
from ui.components.chat import render_message
from ui.components.streaming import start_streaming_turn

_METRICS = {
    "total_time": 2.345,
    "tps": 32.12,
    "input_token_count": 150,
    "token_count": 250,
}


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


def _render_with_metrics(role: str = "assistant") -> MagicMock:
    """render_message를 호출하며 ui.components.chat.st 모듈을 mock 처리합니다."""
    with patch("ui.components.chat.st") as mock_st:
        mock_st.container.return_value.__enter__.return_value = mock_st
        mock_st.chat_message.return_value.__enter__.return_value = mock_st
        mock_st.popover.return_value.__enter__.return_value = mock_st
        render_message(
            role=role,
            content="테스트 답변입니다.",
            metrics=_METRICS,
            wrap_in_container=False,
            is_latest=True,
        )
    return mock_st


def test_metrics_popover_not_rendered():
    """metrics가 있어도 popover/성능 테이블 HTML을 렌더링하지 않습니다."""
    mock_st = _render_with_metrics()

    rendered = "".join(
        str(call.args[0]) if call.args else ""
        for call in mock_st.markdown.call_args_list
    )
    assert '<div class="perf-table">' not in rendered
    assert "성능 지표" not in rendered
    mock_st.popover.assert_not_called()


def test_references_popover_only_with_documents():
    """documents가 없으면 (성능 지표 제거 후) popover가 열리지 않습니다."""
    mock_st = _render_with_metrics()
    mock_st.popover.assert_not_called()


def test_user_role_no_popover_even_with_documents():
    """사용자(user) 역할에는 metrics가 있어도 popover를 렌더링하지 않습니다."""
    doc = MagicMock()
    doc.metadata = {"page": 1}
    with patch("ui.components.chat.st") as mock_st:
        mock_st.container.return_value.__enter__.return_value = mock_st
        render_message(
            role="user",
            content="질문입니다.",
            documents=[doc],
            metrics=_METRICS,
            wrap_in_container=False,
            is_latest=True,
        )
    mock_st.popover.assert_not_called()


def test_streaming_chunk_performance_stored_as_metrics():
    """performance 청크는 렌더링과 무관하게 세션 메시지 metrics에 그대로 저장됩니다."""
    session_id = "metrics_storage"
    SessionManager.reset_all_state(session_id)
    SessionManager.set("is_generating_answer", True, session_id)

    fake_chunks = [
        _fake_chunk(content="답변 내용입니다.", performance=dict(_METRICS)),
    ]
    with patch(
        "ui.components.streaming.stream_chunks",
        return_value=iter(fake_chunks),
    ):
        start_streaming_turn(session_id, "질문", "test-model")

    _wait_for_flag_cleared(session_id)

    messages = SessionManager.get_messages(session_id=session_id)
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["metrics"] == _METRICS
    assert SessionManager.get("is_generating_answer", False, session_id) is False
