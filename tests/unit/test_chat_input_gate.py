"""DG dead-gate TDD RED: chat_input disabled gate + raw-body leakage.

Proves the disabled-input dead gate: ``_resolve_chat_input_state`` knows
the not-ready state, but ``render_chat_input_area`` renders
``st.chat_input`` with ``disabled=False`` and accepts submits without an
``is_ready_for_chat`` guard. Also proves raw ``str(exc)`` leaks into the
assistant body instead of persisting via the ``error=`` field.

Expected: RED on disabled kwarg + raw-body assertions until src/ is fixed.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from common.exceptions import VectorStoreError
from core.session import SessionManager
from ui.components.chat import (
    _friendly_stream_error,
    _render_streaming_with_write_stream,
    _resolve_chat_input_state,
    render_chat_input_area,
)


def _reset_session(sid: str) -> None:
    SessionManager.reset_all_state(sid)
    SessionManager.set_session_id(sid)


def _user_calls(mock_add: MagicMock) -> list[Any]:
    calls: list[Any] = []
    for call in mock_add.call_args_list:
        args = call.args or ()
        kwargs = call.kwargs or {}
        if (args and args[0] == "user") or kwargs.get("role") == "user":
            calls.append(call)
    return calls


def test_resolver_not_ready_returns_guide_and_disabled() -> None:
    """(a) Resolver maps no-file state to (no-PDF placeholder, True)."""
    sid = "dg_gate_resolver"
    _reset_session(sid)

    assert SessionManager.is_ready_for_chat(session_id=sid) is False
    placeholder, disabled = _resolve_chat_input_state(sid)

    assert placeholder == "좌측 사이드바에서 PDF 문서를 먼저 업로드해 주세요."
    assert disabled is True


def test_chat_input_widget_receives_disabled_true_when_not_ready() -> None:
    """(b) RED: widget must receive disabled=True when not-ready."""
    sid = "dg_gate_disabled_kwarg"
    _reset_session(sid)

    with patch("ui.components.chat.st") as mock_st:
        mock_st.chat_input.return_value = None
        render_chat_input_area()

    assert mock_st.chat_input.called, "st.chat_input must be rendered"
    _, kwargs = mock_st.chat_input.call_args
    assert kwargs.get("disabled") is True, (
        f"dead-gate: chat_input disabled={kwargs.get('disabled')!r}, "
        "expected True when not-ready"
    )


def test_submit_refused_when_not_ready_no_user_message() -> None:
    """(c) RED: submit while not-ready must not persist a user message."""
    sid = "dg_gate_submit_refused"
    _reset_session(sid)

    with (
        patch("ui.components.chat.st") as mock_st,
        patch.object(SessionManager, "add_message") as mock_add,
    ):
        mock_st.chat_input.return_value = "hello dead-gate probe"
        render_chat_input_area()

    assert _user_calls(mock_add) == [], (
        "dead-gate: user submit persisted while is_ready_for_chat=False"
    )


def test_submit_refused_guides_via_error_not_body_no_rerun() -> None:
    """(c) RED: not-ready submit guides via st.error, no body, no rerun."""
    sid = "dg_gate_submit_guide"
    _reset_session(sid)
    expected_placeholder = "좌측 사이드바에서 PDF 문서를 먼저 업로드해 주세요."

    with (
        patch("ui.components.chat.st") as mock_st,
        patch.object(SessionManager, "add_message") as mock_add,
    ):
        mock_st.chat_input.return_value = "hello dead-gate probe"
        render_chat_input_area()

    mock_st.error.assert_called_once()
    err_text = str(mock_st.error.call_args)
    assert expected_placeholder in err_text, (
        f"dead-gate: st.error must show no-PDF placeholder, got {err_text!r}"
    )
    for call in mock_add.call_args_list:
        args = call.args or ()
        kwargs = call.kwargs or {}
        content = str(args[1]) if len(args) > 1 else str(kwargs.get("content"))
        assert expected_placeholder not in content, (
            "dead-gate: guide must go via st.error, not message body"
        )
    mock_st.rerun.assert_not_called()


def test_vector_store_error_persists_error_field_without_raw_body() -> None:
    """(d) RED: VectorStoreError persists error=, body hides raw str(exc)."""
    sid = "dg_gate_raw_body"
    _reset_session(sid)
    sentinel = "DG_RAW_SENTINEL_9f3a"
    exc = VectorStoreError(reason=sentinel)
    assert sentinel in str(exc)

    friendly = _friendly_stream_error(exc)
    assert sentinel not in friendly, "friendly message must not leak raw text"

    msg_id = "dg-stream-mid-1"
    with (
        patch("ui.components.chat.st") as mock_st,
        patch(
            "ui.components.chat.render_generation_expander",
            return_value=None,
        ),
    ):
        mock_st.chat_message.return_value.__enter__.return_value = MagicMock()
        mock_st.chat_message.return_value.__exit__.return_value = False
        mock_st.empty.return_value = MagicMock()
        mock_st.write_stream.side_effect = exc
        _render_streaming_with_write_stream(
            {"msg_id": msg_id, "thought": "", "documents": []},
            sid,
            "probe query",
        )

    persisted = [
        m for m in SessionManager.get("messages", [], sid) if m.get("msg_id") == msg_id
    ]
    assert persisted, "streaming turn must persist the assistant message"
    body = str(persisted[-1].get("content") or "")
    assert sentinel not in body, f"dead-gate: raw str(exc) leaked into body: {body!r}"
    assert persisted[-1].get("error"), (
        "dead-gate: VectorStoreError must persist via error= field"
    )
