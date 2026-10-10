"""P2: 문서 분석 카드 렌더 테스트 (BuildState 단일 진실원천, 3B)."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from core.build_state import (
    BuildPhase,
    complete_build,
    fail_build,
    initial_build_state,
)
from core.session import SessionManager


@contextmanager
def _mock_st():
    import ui.components.doc_analysis.panel as panel_mod

    with patch.object(panel_mod, "st") as mock_st:
        mock_st.status.return_value.__enter__.return_value = MagicMock()
        mock_st.button.return_value = False
        yield mock_st


def _sid(name: str) -> str:
    sid = f"test-build-card-{name}"
    SessionManager.reset_all_state(sid)
    SessionManager.set_session_id(sid)
    return sid


def test_running_card_shows_stepper_and_progress() -> None:
    from ui.components import chat_build as build_mod

    sid = _sid("running")
    build = initial_build_state("doc.pdf", "h1")
    build.phase = BuildPhase.INDEXING
    build.pct = 68
    SessionManager.set("build_state", build, session_id=sid)
    SessionManager.set("is_building_rag", True, session_id=sid)

    with _mock_st() as mock_st:
        build_mod._render_build_progress_block(sid)

    mock_st.progress.assert_called_once_with(0.68)
    captions = [c.args[0] for c in mock_st.caption.call_args_list]
    assert any("▸" in cap and "✓" in cap for cap in captions), captions
    assert any("68" in cap for cap in captions), captions
    # 실행 중에는 접히지 않는다
    assert mock_st.status.call_args.kwargs["expanded"] is True
    assert mock_st.status.call_args.kwargs["state"] == "running"


def test_done_card_collapsed_without_progress() -> None:
    from ui.components import chat_build as build_mod

    sid = _sid("done")
    build = initial_build_state("doc.pdf", "h1")
    complete_build(build, cache_used=True)
    SessionManager.set("build_state", build, session_id=sid)

    with _mock_st() as mock_st:
        build_mod._render_build_progress_block(sid)

    assert mock_st.status.call_args.kwargs["expanded"] is False
    assert mock_st.status.call_args.kwargs["state"] == "complete"
    mock_st.progress.assert_not_called()
    label = mock_st.status.call_args.args[0]
    assert "기존 분석 재사용" in label or "reusing" in label, label


def test_error_card_offers_retry() -> None:
    from ui.components import chat_build as build_mod

    sid = _sid("error")
    build = initial_build_state("doc.pdf", "h1")
    fail_build(build, "boom")
    SessionManager.set("build_state", build, session_id=sid)

    with _mock_st() as mock_st:
        build_mod._render_build_progress_block(sid)

    assert mock_st.status.call_args.kwargs["state"] == "error"
    assert mock_st.button.called


def test_no_build_state_renders_nothing() -> None:
    """BuildState 없이는 아무것도 렌더하지 않는다 (3B: 레거시 폴백 삭제)."""
    from ui.components import chat_build as build_mod

    sid = _sid("none")
    SessionManager.set("is_building_rag", True, session_id=sid)
    SessionManager.set("last_uploaded_file_name", "old.pdf", session_id=sid)

    with _mock_st() as mock_st:
        build_mod._render_build_progress_block(sid)

    mock_st.status.assert_not_called()
    mock_st.progress.assert_not_called()


def test_cancelled_card_is_terminal_error_with_retry() -> None:
    """CANCELLED는 terminal error + 재시도 (2B). 스테퍼/스피너 없음."""
    from ui.components import chat_build as build_mod

    sid = _sid("cancelled")
    build = initial_build_state("doc.pdf", "h1")
    build.phase = BuildPhase.CANCELLED
    build.pct = 40
    SessionManager.set("build_state", build, session_id=sid)

    with _mock_st() as mock_st:
        build_mod._render_build_progress_block(sid)

    assert mock_st.status.call_args.kwargs["state"] == "error"
    assert mock_st.status.call_args.kwargs["expanded"] is True
    mock_st.progress.assert_not_called()
    assert mock_st.button.called


def test_stepper_caption_marks() -> None:
    from ui.components.doc_analysis.viewmodel import stepper_caption

    cap = stepper_caption(BuildPhase.CHUNKING)
    assert cap is not None
    assert cap.count("✓") == 1
    assert "▸" in cap
    assert "○" in cap
    assert stepper_caption(BuildPhase.DONE) is None
    assert stepper_caption(BuildPhase.ERROR) is None
    assert stepper_caption(BuildPhase.CANCELLED) is None


def test_no_build_state_no_file_renders_nothing() -> None:
    from ui.components import chat_build as build_mod

    sid = _sid("empty")
    with _mock_st() as mock_st:
        build_mod._render_build_progress_block(sid)
    mock_st.status.assert_not_called()
    mock_st.progress.assert_not_called()
