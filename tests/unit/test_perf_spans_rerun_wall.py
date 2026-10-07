"""Task 0 instrumentation gate (TDD failing-first): post-stream [PERF] spans.

Makes post-stream latency measurable BEFORE any optimization:

- ``rerun-wall`` must span the actual Streamlit rerun (pre-rerun timestamp
  persisted via ``SessionManager`` before ``st.rerun()``, measured on the
  next script run) — not the ~0ms ``st.rerun()`` call time that
  ``chat.py`` used to log (``_t_rerun=perf_counter(); log(...); st.rerun()``).
- ``last-token->citations-event`` span must exist with ``msg_id``.
- ``turn_start`` baseline must be recorded at submit time with ``msg_id``.

Logging only — no behavior change beyond ``[PERF]`` timers.
"""

from __future__ import annotations

import logging
import re
import time
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import ui.components.chat as chat_mod
import ui.components.streaming_core as sc_mod
from core.session import SessionManager


class _RerunSignal(BaseException):
    """Stand-in for Streamlit's RerunException (a ``BaseException``)."""


def _fake_stream_ui() -> MagicMock:
    """Minimal ``st`` double: real rerun semantics (raises), inert widgets."""
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
    fake_st.rerun.side_effect = _RerunSignal()
    return fake_st


def _run_tail_to_rerun(sid: str, msg_id: str) -> MagicMock:
    """Drive a STOPPED turn tail until ``st.rerun()`` raises; return ``st``.

    Clean turns skip ``st.rerun()`` since Task3 (terminalize guard), so the
    rerun-wall span is only measurable on turns that still rerun
    (■ stop / error). ``generation_cancel`` is set before driving and the
    tail itself resets it after persisting (read before reset).
    """
    import pytest

    SessionManager.set("generation_cancel", True, session_id=sid)
    fake_st = _fake_stream_ui()
    with (
        patch.object(chat_mod, "st", fake_st),
        patch.object(chat_mod, "_content_generator", return_value=iter(["x"])),
        patch.object(chat_mod, "_clear_aux_state", return_value=None),
        patch.object(chat_mod, "_finalize_pdf_side_effects", return_value=None),
        patch.object(chat_mod, "render_generation_expander", return_value=None),
        pytest.raises(_RerunSignal),
    ):
        chat_mod._render_streaming_with_write_stream(
            {"msg_id": msg_id}, sid, "wall query"
        )
    return fake_st


def _run_clean_tail(sid: str, msg_id: str) -> MagicMock:
    """Drive a clean turn tail; must NOT rerun (Task3 skip contract)."""
    fake_st = _fake_stream_ui()
    with (
        patch.object(chat_mod, "st", fake_st),
        patch.object(chat_mod, "_content_generator", return_value=iter(["x"])),
        patch.object(chat_mod, "_clear_aux_state", return_value=None),
        patch.object(chat_mod, "_finalize_pdf_side_effects", return_value=None),
        patch.object(chat_mod, "render_generation_expander", return_value=None),
    ):
        chat_mod._render_streaming_with_write_stream(
            {"msg_id": msg_id}, sid, "wall query"
        )
    return fake_st


def test_clean_turn_skips_rerun(session_context: str, caplog: Any) -> None:
    """Task3 contract: clean finalize paints in-run, so no rerun + no wall."""
    sid = session_context
    msg_id = "msg-wall-clean-1"

    with caplog.at_level(logging.INFO, logger="ui.components.chat"):
        caplog.clear()
        fake_st = _run_clean_tail(sid, msg_id)

    assert fake_st.rerun.call_count == 0
    assert SessionManager.get("stream_finalized_no_rerun", None, sid) == msg_id
    assert f"[PERF] rerun skipped msg_id={msg_id}" in caplog.text
    # No wall pending — the next script run must not log a stale wall.
    assert SessionManager.get("perf_rerun_start", None, sid) is None


def test_rerun_wall_spans_actual_rerun(session_context: str, caplog: Any) -> None:
    """RED: wall is measured across the rerun via a persisted timestamp."""
    sid = session_context
    msg_id = "msg-wall-1"

    fake_st = _run_tail_to_rerun(sid, msg_id)
    assert fake_st.rerun.call_count == 1

    # Pre-rerun clock must survive st.rerun() (real rerun discards locals).
    rerun_start = SessionManager.get("perf_rerun_start", None, sid)
    assert rerun_start is not None, "pre-rerun timestamp must be persisted"
    assert SessionManager.get("perf_rerun_msg_id", None, sid) == msg_id

    # Simulate the next script run after genuine wall time has passed.
    time.sleep(0.02)
    with caplog.at_level(logging.INFO, logger="ui.components.chat"):
        caplog.clear()
        chat_mod.report_rerun_wall_if_pending(sid)

    match = re.search(r"\[PERF\] rerun-wall msg_id=(\S+) took ([\d.]+)s", caplog.text)
    assert match is not None, "rerun-wall span with msg_id must be logged"
    assert match.group(1) == msg_id
    assert float(match.group(2)) >= 0.015, (
        "wall must span the rerun, not the ~0ms st.rerun() call time"
    )
    # Consumed exactly once — a second run must not re-log a stale wall.
    assert SessionManager.get("perf_rerun_start", None, sid) is None


def test_last_token_to_citations_span_exists(session_context: str, caplog: Any) -> None:
    """RED: last-token -> citations-event span exists with msg_id."""
    sid = session_context
    msg_id = "msg-cite-1"

    def _chunk(**kw: Any) -> Any:
        base: dict[str, Any] = {
            "status": None,
            "content": None,
            "thought": None,
            "metadata": {},
            "performance": None,
            "citations": None,
            "raw_json": False,
        }
        base.update(kw)
        return SimpleNamespace(**base)

    chunks = [
        _chunk(content="hello "),
        _chunk(citations=[{"doc_id": "doc-0"}]),
    ]
    with (
        patch.object(sc_mod, "stream_chunks", return_value=iter(chunks)),
        patch.object(sc_mod, "_note_reference_prewarm", return_value=None),
        caplog.at_level(logging.INFO, logger="ui.components.streaming_core"),
    ):
        caplog.clear()
        out = list(sc_mod._content_generator("q", "m", sid, msg_id))

    assert out == ["hello "]
    match = re.search(
        r"\[PERF\] last-token->citations-event msg_id=(\S+) took ([\d.]+)s",
        caplog.text,
    )
    assert match is not None, "last-token->citations-event span must be logged"
    assert match.group(1) == msg_id
    assert float(match.group(2)) >= 0.0


def test_turn_start_baseline_recorded(session_context: str, caplog: Any) -> None:
    """RED: submit-time turn_start is persisted and surfaced as elapsed."""
    sid = session_context
    msg_id = "msg-turn-1"

    t0 = time.perf_counter()
    chat_mod._record_turn_start(sid, msg_id, t0)
    assert SessionManager.get("perf_turn_start", None, sid) == t0
    assert SessionManager.get("perf_turn_msg_id", None, sid) == msg_id

    with caplog.at_level(logging.INFO, logger="ui.components.chat"):
        caplog.clear()
        _run_tail_to_rerun(sid, msg_id)

    match = re.search(r"\[PERF\] turn msg_id=(\S+) elapsed_sec=([\d.]+)s", caplog.text)
    assert match is not None, "turn elapsed baseline must be logged"
    assert match.group(1) == msg_id
    assert float(match.group(2)) >= 0.0
