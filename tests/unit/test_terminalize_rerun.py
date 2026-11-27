"""Task3 terminalize: streaming finalize must not require st.rerun.

단순안: clean finalize는 tail에서 참조 목록 1회만 in-run 확정 렌더하고
``st.rerun()`` 을 스킵한다 (익스팬더/배지/지표 없음). Stop/error turns still
rerun for a confirmed render.
"""

from __future__ import annotations

import inspect
import uuid
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import pytest

import ui.components.chat as chat_module
from core.session import SessionManager


def _make_st_mock() -> MagicMock:
    """Bare-bones Streamlit stand-in (context managers + write_stream work)."""
    st_mock = MagicMock(name="st")
    st_mock.session_state = {}
    st_mock.write_stream.side_effect = lambda gen: "".join(gen)
    return st_mock


def _fake_content_generator(
    query: str,
    model_name: str,
    sid: str,
    mid: str,
    on_status: Any = None,
    on_aux: Any = None,
) -> Iterator[str]:
    """Pretend the stream delivered citations, then one text chunk."""
    from ui.components.streaming import _AUX_STATE_KEY

    aux: dict[str, Any] = {
        "thought": "t",
        "documents": [],
        "metrics": {},
        "citations": [{"doc_id": "d1", "page": 1}],
        "process_steps": [],
        "complete": False,
        "content": "hello",
    }
    SessionManager.set(_AUX_STATE_KEY, aux, sid)
    if on_aux is not None:
        on_aux(aux)
    yield "hello"


def _failing_content_generator(
    query: str,
    model_name: str,
    sid: str,
    mid: str,
    on_status: Any = None,
    on_aux: Any = None,
) -> Iterator[str]:
    raise RuntimeError("boom")
    yield "unreachable"


def _run_finalize(
    monkeypatch: pytest.MonkeyPatch,
    sid: str,
    mid: str,
    *,
    cancel: bool = False,
    boom: bool = False,
) -> tuple[MagicMock, MagicMock]:
    """Drive the real finalize tail with Streamlit + side effects mocked."""
    st_mock = _make_st_mock()
    monkeypatch.setattr(chat_module, "st", st_mock)
    monkeypatch.setattr(
        chat_module,
        "_content_generator",
        _failing_content_generator if boom else _fake_content_generator,
    )
    monkeypatch.setattr(chat_module, "render_reference_list", MagicMock(name="refs"))
    refs_mock = chat_module.render_reference_list
    monkeypatch.setattr(
        chat_module, "_finalize_pdf_side_effects", MagicMock(name="pdf_fx")
    )
    if cancel:
        SessionManager.set("generation_cancel", True, sid)
    msg = {"msg_id": mid, "query": "q", "content": ""}
    chat_module._render_streaming_with_write_stream(msg, sid, "q")
    return st_mock, refs_mock


def test_clean_finalize_skips_rerun(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clean turn: msg_type conversion alone must not cost an st.rerun."""
    sid = f"task3_{uuid.uuid4().hex[:8]}"
    mid = f"msg_{uuid.uuid4().hex[:8]}"
    st_mock, _ = _run_finalize(monkeypatch, sid, mid)
    st_mock.rerun.assert_not_called()
    assert SessionManager.get("stream_finalized_no_rerun", None, sid) == mid


def test_live_body_repaints_without_context_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed turn must not leave raw [doc:...] tokens in the live bubble."""
    from ui.components.streaming import _AUX_STATE_KEY

    sid = f"task3_{uuid.uuid4().hex[:8]}"
    mid = f"msg_{uuid.uuid4().hex[:8]}"
    raw = "answer [doc:d1] here"

    st_mock = _make_st_mock()
    placeholders = [MagicMock(name=f"ph{i}") for i in range(3)]
    st_mock.empty.side_effect = placeholders
    monkeypatch.setattr(chat_module, "st", st_mock)

    def token_generator(
        query: str,
        model_name: str,
        session_id: str,
        message_id: str,
        on_status: Any = None,
        on_aux: Any = None,
    ) -> Iterator[str]:
        aux: dict[str, Any] = {
            "content": raw,
            "citations": [],
            "documents": [],
            "metrics": {},
            "process_steps": [],
        }
        SessionManager.set(_AUX_STATE_KEY, aux, session_id)
        if on_aux is not None:
            on_aux(aux)
        yield raw

    monkeypatch.setattr(chat_module, "_content_generator", token_generator)
    monkeypatch.setattr(chat_module, "render_reference_list", MagicMock())
    monkeypatch.setattr(chat_module, "_finalize_pdf_side_effects", MagicMock())

    chat_module._render_streaming_with_write_stream({"msg_id": mid}, sid, "q")

    body_ph = placeholders[2]
    body_ph.markdown.assert_called_once()
    rendered = body_ph.markdown.call_args.args[0]
    assert "[doc:" not in rendered
    assert "answer" in rendered
    st_mock.rerun.assert_not_called()


def test_clean_finalize_paints_refs_in_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """단순안: tail에서 참조 목록 1회만 in-run 확정 렌더한다."""
    sid = f"task3_{uuid.uuid4().hex[:8]}"
    mid = f"msg_{uuid.uuid4().hex[:8]}"
    _, refs_mock = _run_finalize(monkeypatch, sid, mid)
    refs_mock.assert_called_once()


def test_cancelled_finalize_still_reruns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop turn: partial answer needs the confirmed rerun render."""
    sid = f"task3_{uuid.uuid4().hex[:8]}"
    mid = f"msg_{uuid.uuid4().hex[:8]}"
    st_mock, _ = _run_finalize(monkeypatch, sid, mid, cancel=True)
    st_mock.rerun.assert_called_once()


def test_error_finalize_still_reruns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Error turn: surfaced error needs the confirmed rerun render."""
    sid = f"task3_{uuid.uuid4().hex[:8]}"
    mid = f"msg_{uuid.uuid4().hex[:8]}"
    st_mock, _ = _run_finalize(monkeypatch, sid, mid, boom=True)
    st_mock.rerun.assert_called_once()


def test_should_skip_final_rerun_helper() -> None:
    """Guard truth table: only clean turns skip the terminal rerun."""
    assert (
        chat_module._should_skip_final_rerun(cancelled=False, error_text=None) is True
    )
    assert (
        chat_module._should_skip_final_rerun(cancelled=True, error_text=None) is False
    )
    assert (
        chat_module._should_skip_final_rerun(cancelled=False, error_text="boom")
        is False
    )


def test_streaming_finalize_never_calls_refs_body_directly() -> None:
    """Refs body renders once via the final expander, not per history item."""
    source = inspect.getsource(chat_module._render_streaming_with_write_stream)
    assert "_render_references_content" not in source
