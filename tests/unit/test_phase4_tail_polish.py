"""Phase 4: metrics/error/bridge finishing guards.

- Error turns keep the evidence expander (thought/documents/citations stay
  reachable) and prefix the error via ``t("error_prefix")``.
- Tooltip cache signature is value-based: re-rendering the same content with
  reconstructed (non-identical) document lists hits the cache instead of
  rescanning every rerun.
- ``is_latest`` follows the last *visible* message: trailing internal
  messages (``READY_FOR_QUERY``) no longer collapse the real last answer.
- ``UIBridge.sync_session`` forces one sync on session switch even without
  pending dirty keys (first-render stale mirror fix); same-session skips stay.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import ui.strings as strings_mod
from ui.strings import set_lang, t


def _doc(page: int = 1) -> MagicMock:
    doc = MagicMock()
    doc.metadata = {"page": page}
    doc.page_content = "참조 본문"
    return doc


def test_error_turn_preserves_evidence_expander() -> None:
    """error + thought/documents -> expander rendered before the error card."""
    import ui.components.chat as chat_mod

    with (
        patch.object(chat_mod, "st") as mock_st,
        patch.object(chat_mod, "render_generation_expander") as mock_exp,
        patch("ui.components.common.st", mock_st),
    ):
        mock_st.chat_message.return_value.__enter__.return_value = MagicMock()
        mock_st.chat_message.return_value.__exit__.return_value = False
        chat_mod.render_message(
            role="assistant",
            content="부분 답변",
            thought="추론",
            documents=[_doc(2)],
            error="boom",
        )
    assert mock_exp.called, "evidence expander must survive error turns"
    mock_st.error.assert_called_once()


def test_error_prefix_uses_i18n_key() -> None:
    """Error card prefix comes from t('error_prefix') in the active locale."""
    import ui.components.chat as chat_mod

    original = strings_mod.LANG
    set_lang("ko")
    try:
        expected = t("error_prefix")
        with (
            patch.object(chat_mod, "st") as mock_st,
            patch.object(chat_mod, "render_generation_expander"),
            patch("ui.components.common.st", mock_st),
        ):
            mock_st.chat_message.return_value.__enter__.return_value = MagicMock()
            mock_st.chat_message.return_value.__exit__.return_value = False
            chat_mod.render_message(role="assistant", content="본문", error="E1")
    finally:
        set_lang(original)
    err_text = str(mock_st.error.call_args.args[0])
    assert err_text.startswith(expected)
    assert "E1" in err_text


def test_tooltip_cache_hits_on_reconstructed_lists() -> None:
    """Same content + equal (not identical) docs -> single tooltip scan."""
    import ui.components.chat as chat_mod

    state: dict[str, Any] = {}
    fake_st = MagicMock()
    fake_st.session_state = state
    fake_st.chat_message.return_value.__enter__.return_value = MagicMock()
    fake_st.chat_message.return_value.__exit__.return_value = False

    kwargs: dict[str, Any] = {
        "role": "assistant",
        "content": "답변 본문",
        "documents": None,
        "msg_id": "tip-1",
        "msg_index": 0,
        "is_latest": False,
    }
    with (
        patch.object(chat_mod, "st", fake_st),
        patch.object(
            chat_mod, "apply_tooltips_to_response", return_value="SCANNED"
        ) as mock_scan,
        patch.object(chat_mod, "render_generation_expander"),
    ):
        kwargs["documents"] = [_doc(3)]
        chat_mod.render_message(**kwargs)
        # Reconstructed equal list (different object identities, as on rerun).
        kwargs["documents"] = [_doc(3)]
        chat_mod.render_message(**kwargs)
    assert mock_scan.call_count == 1


def test_trailing_internal_message_keeps_last_answer_latest() -> None:
    """[assistant, READY_FOR_QUERY] -> assistant turn still gets is_latest."""
    import ui.components.chat as chat_mod

    messages = [
        {
            "role": "assistant",
            "content": "답변",
            "msg_type": "general",
            "msg_id": "a1",
            "thought": "추론",
            "documents": [],
        },
        {"role": "user", "content": "READY_FOR_QUERY", "msg_type": "general"},
    ]
    with (
        patch.object(chat_mod.SessionManager, "get_messages", return_value=messages),
        patch.object(chat_mod.SessionManager, "get", return_value=False),
        patch.object(chat_mod, "render_message") as mock_render,
    ):
        chat_mod._render_unified_timeline("sid-tail")
    assert mock_render.called
    assert mock_render.call_args.kwargs.get("is_latest") is True


def test_bridge_forces_sync_on_session_switch() -> None:
    """New sid without dirty keys still syncs once; repeats still skip."""
    import ui.bridge as bridge_mod
    from ui.bridge import UIBridge

    bridge_mod._last_synced_sid = None
    with (
        patch.object(
            bridge_mod.SessionManager, "get_session_id", return_value="sid-new"
        ),
        patch.object(
            bridge_mod.SessionManager, "has_pending_ui_sync", return_value=False
        ),
        patch.object(bridge_mod.SessionManager, "sync_to_streamlit") as mock_sync,
    ):
        UIBridge.sync_session()
        assert mock_sync.call_count == 1
        UIBridge.sync_session()
        assert mock_sync.call_count == 1
    assert bridge_mod._last_synced_sid == "sid-new"
