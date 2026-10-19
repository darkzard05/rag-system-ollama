from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest

from core.session import SessionManager
import ui.components.chat as chat_mod
from ui.components.streaming import consume_stream_into_message


def _fake_chunk(content: str = "", status: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        status=status,
        thought="",
        content=content,
        metadata=None,
        performance=None,
    )


def test_streaming_persists_processed_content():
    """스트리밍 완료 시 processed_content가 None이 아닌 사전 가공된 마크다운 텍스트로 저장되는지 검증."""
    sid = "test_processed_cache_sid"
    SessionManager.reset_all_state(sid)

    raw_text = "이것은 [doc:dummy123] [page:1] 수식 \(x^2\) 테스트입니다 & <특수문자>"
    fake_chunks = [_fake_chunk(content=raw_text)]

    with patch("ui.components.streaming.stream_chunks", return_value=iter(fake_chunks)):
        result = consume_stream_into_message(sid, "질문", "test-model")

    assert result is not None
    messages = SessionManager.get_messages(session_id=sid)
    target = next(m for m in messages if m.get("msg_id") == result["msg_id"])

    # 1. 원본 content는 보존되어야 함
    assert target["content"] == raw_text
    # 2. processed_content가 사전 계산되어 저장되어야 함
    processed = target.get("processed_content")
    assert processed is not None
    # 3. HTML escape 확인
    assert "&amp;" in processed or "&lt;특수문자&gt;" in processed
    # 4. LaTeX 정규화 확인 (\( -> $)
    assert "$x^2$" in processed
    # 5. 컨텍스트 토큰 제거 확인
    assert "[doc:dummy123]" not in processed


def test_render_message_uses_processed_content_fast_path():
    """processed_content가 존재할 때 불필요한 LaTeX/컨텍스트 토큰 정규식을 재실행하지 않고 즉시 렌더링하는지 검증."""
    mock_st = MagicMock()
    mock_st.container.return_value.__enter__.return_value = mock_st
    mock_st.chat_message.return_value.__enter__.return_value = mock_st

    precomputed = "<p>Fast Path Precomputed Content</p>"

    with (
        patch("ui.components.chat.st", mock_st),
        patch("ui.components.chat.normalize_latex_delimiters") as mock_latex,
        patch("ui.components.chat.strip_context_tokens") as mock_strip,
    ):
        chat_mod.render_message(
            role="assistant",
            content="Original Unprocessed Content",
            processed_content=precomputed,
            msg_id="m_fast",
            wrap_in_container=False,
        )

    # Fast Path: 정규식 함수들이 전혀 호출되지 않아야 함
    mock_latex.assert_not_called()
    mock_strip.assert_not_called()

    # st.markdown에는 사전 계산된 본문이 즉시 전달되어야 함
    markdown_calls = [c.args[0] for c in mock_st.markdown.call_args_list if c.args]
    assert any(precomputed in text for text in markdown_calls)
