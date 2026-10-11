"""UX-4: answer-error turns render a retry CTA resubmitting the last query."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import ui.components.chat as chat_mod


class _StubSession:
    def __init__(self, messages: list[dict[str, Any]], generating: bool = False):
        self._messages = messages
        self._generating = generating

    def get_messages(self, **kwargs: Any) -> list[dict[str, Any]]:
        return [dict(m) for m in self._messages]

    def get_session_id(self) -> str:
        return "sid-retry"

    def get(self, key: str, default: Any = None, *args: Any, **kwargs: Any) -> Any:
        if key == "is_generating_answer":
            return self._generating
        return default


def _render_error(messages: list[dict[str, Any]], error: Any, generating: bool = False):
    mock_st = MagicMock()
    mock_st.chat_message.return_value.__enter__.return_value = MagicMock()
    mock_st.chat_message.return_value.__exit__.return_value = False
    with (
        patch.object(chat_mod, "st", mock_st),
        patch.object(chat_mod, "SessionManager", _StubSession(messages, generating)),
        patch("ui.components.common.st", mock_st),
    ):
        chat_mod.render_message(
            role="assistant", content="partial", msg_id="err-1", error=error
        )
    return mock_st


def _button_keys(mock_st) -> list[str]:
    keys = []
    for call in mock_st.button.call_args_list:
        keys.append(str(call.kwargs.get("key", "")))
    return keys


def test_error_with_prior_query_renders_retry() -> None:
    """오류 + 직전 사용자 질문 → retry_error 키의 재시도 버튼."""
    mock_st = _render_error(
        [{"role": "user", "content": "hello?"}, {"role": "assistant", "content": "x"}],
        "boom",
    )
    assert "retry_error_err-1" in _button_keys(mock_st)


def test_error_non_retryable_skips_retry() -> None:
    """retryable=False 오류(FriendlyError) → 버튼 없음."""
    from ui.components.streaming import FriendlyError

    mock_st = _render_error(
        [{"role": "user", "content": "hello?"}],
        FriendlyError(message="stopped", retryable=False),
    )
    assert _button_keys(mock_st) == []


def test_error_without_query_skips_retry() -> None:
    """직전 질문 없음 → 버튼 없음."""
    mock_st = _render_error([], "boom")
    assert _button_keys(mock_st) == []


def test_error_while_generating_skips_retry() -> None:
    """생성 중 렌더된 과거 오류 턴 → 버튼 없음(진행 중 스트림 보호)."""
    mock_st = _render_error(
        [{"role": "user", "content": "hello?"}], "boom", generating=True
    )
    assert _button_keys(mock_st) == []
