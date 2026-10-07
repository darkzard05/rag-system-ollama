"""Tail timing spans TDD: rerun + recompute [PERF] markers.

Wave1 instrumentation gate (additive timers/logs only):
- ``_render_streaming_with_write_stream`` emits ``[PERF] rerun`` once/turn.
- ``_render_references_content`` emits ``[PERF] recompute`` breakdown.
- Ordering invariant: persist BEFORE finalize (untouched, asserted here).
"""

from __future__ import annotations

import inspect
import logging
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def _make_docs(n: int = 2) -> list[dict[str, Any]]:
    """Minimal dict-docs supported by doc_stable_id/get_doc_metadata."""
    return [
        {
            "page_content": f"tail perf doc body {i} alpha beta gamma.",
            "metadata": {
                "doc_id": f"doc-{i}",
                "page": i + 1,
                "current_section": f"Section {i}",
            },
        }
        for i in range(n)
    ]


def test_recompute_span_emitted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """RED: recompute breakdown log present with doc/citation counts."""
    from ui.components import chat_references as refs

    fake_st = MagicMock()
    with (
        patch.object(refs, "st", fake_st),
        caplog.at_level(logging.INFO, logger="ui.components.chat_references"),
    ):
        caplog.clear()
        ok = refs._render_references_content(
            "msg-tail-1",
            _make_docs(2),
            citations=None,
            generating=True,
        )
    assert ok is True
    assert "[PERF] recompute" in caplog.text


def test_recompute_signature_compatible() -> None:
    """Guard: additive timer must not change the public signature."""
    from ui.components import chat_references as refs

    sig = inspect.signature(refs._render_references_content)
    assert list(sig.parameters) == [
        "msg_id",
        "documents",
        "citations",
        "generating",
        "show_caption",
    ]


def test_rerun_span_emitted_once_per_turn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Task3 amendment: a CLEAN tail skips rerun (final painted in-run).

    The ``[PERF] rerun`` marker is still emitted exactly once — as the
    ``rerun skipped`` variant — and persist-before-finalize ordering holds.
    ■ stop / error turns still rerun (see ``test_stopped_turn_reruns_once``).
    """
    from ui.components import chat as chat_mod

    fake_st = MagicMock()
    fake_st.write_stream.return_value = "hello tail"

    @contextmanager
    def _fake_chat_message(*args: Any, **kwargs: Any) -> Any:
        yield MagicMock()

    fake_st.chat_message.side_effect = _fake_chat_message
    ph = MagicMock()
    ph.empty.return_value = None
    ph.caption.return_value = None
    fake_st.empty.return_value = ph

    with (
        patch.object(chat_mod, "st", fake_st),
        patch.object(chat_mod, "_content_generator", return_value=iter(["hello "])),
        patch.object(chat_mod, "_clear_aux_state", return_value=None),
        patch.object(chat_mod, "_finalize_pdf_side_effects", return_value=None),
        patch.object(chat_mod, "render_generation_expander", return_value=None),
        patch.object(chat_mod.SessionManager, "get", return_value=""),
        patch.object(chat_mod.SessionManager, "add_message", return_value=None),
        patch.object(chat_mod.SessionManager, "set", return_value=None),
        caplog.at_level(logging.INFO, logger="ui.components.chat"),
    ):
        caplog.clear()
        chat_mod._render_streaming_with_write_stream(
            {"msg_id": "msg-tail-9"},
            "sid-tail",
            "tail query",
        )

    assert fake_st.rerun.call_count == 0
    assert caplog.text.count("[PERF] rerun") == 1
    assert "[PERF] rerun skipped msg_id=msg-tail-9" in caplog.text
    assert "[PERF] live persist" in caplog.text
    assert "[PERF] finalize" in caplog.text
    assert caplog.text.find("[PERF] live persist") < caplog.text.find("[PERF] finalize")


def test_stopped_turn_reruns_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """■ stop turns still rerun exactly once for a confirmed render."""
    from ui.components import chat as chat_mod

    fake_st = MagicMock()
    fake_st.write_stream.return_value = "hello tail"

    @contextmanager
    def _fake_chat_message(*args: Any, **kwargs: Any) -> Any:
        yield MagicMock()

    fake_st.chat_message.side_effect = _fake_chat_message
    ph = MagicMock()
    ph.empty.return_value = None
    ph.caption.return_value = None
    fake_st.empty.return_value = ph

    def _get(key: str, default: Any = None, *args: Any, **kwargs: Any) -> Any:
        if key == "generation_cancel":
            return True
        return ""

    with (
        patch.object(chat_mod, "st", fake_st),
        patch.object(chat_mod, "_content_generator", return_value=iter(["hello "])),
        patch.object(chat_mod, "_clear_aux_state", return_value=None),
        patch.object(chat_mod, "_finalize_pdf_side_effects", return_value=None),
        patch.object(chat_mod, "render_generation_expander", return_value=None),
        patch.object(chat_mod.SessionManager, "get", side_effect=_get),
        patch.object(chat_mod.SessionManager, "add_message", return_value=None),
        patch.object(chat_mod.SessionManager, "set", return_value=None),
        caplog.at_level(logging.INFO, logger="ui.components.chat"),
    ):
        caplog.clear()
        chat_mod._render_streaming_with_write_stream(
            {"msg_id": "msg-tail-stop-1"},
            "sid-tail",
            "tail query",
        )

    assert fake_st.rerun.call_count == 1
    assert "[PERF] rerun scheduled msg_id=msg-tail-stop-1" in caplog.text
