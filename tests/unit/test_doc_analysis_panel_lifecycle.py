"""문서 분석 패널(st.status) 라이프사이클 및 플리커링 방지 검증 단위 테스트."""

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
from ui.components.doc_analysis.panel import render_document_panel


@contextmanager
def _mock_st_status_container():
    import ui.components.doc_analysis.panel as panel_mod

    mock_container = MagicMock()
    with (
        patch.object(panel_mod, "st") as mock_st,
        patch.object(panel_mod, "ui_error"),
    ):
        mock_st.status.return_value.__enter__.return_value = mock_container
        mock_st.status.return_value.__exit__.return_value = None
        mock_st.button.return_value = False
        yield mock_st, mock_container


def test_preparing_panel_is_compact_until_extracting():
    """PREPARING은 접힌 카드(stepper/progress 없음)로 뷰어 선행성을 보장한다."""
    sid = "test-panel-preparing"
    SessionManager.reset_all_state(sid)

    build = initial_build_state("doc.pdf", "hash1")
    SessionManager.set("build_state", build, session_id=sid)
    SessionManager.set("is_building_rag", True, session_id=sid)

    with _mock_st_status_container() as (mock_st, container):
        render_document_panel(sid)

    assert mock_st.status.call_args.kwargs["expanded"] is False
    assert mock_st.status.call_args.kwargs["state"] == "running"
    container.update.assert_called_once()
    assert container.update.call_args.kwargs["expanded"] is False
    assert container.update.call_args.kwargs["state"] == "running"


def test_running_panel_explicitly_updates_expanded_true():
    """running 상태는 status_container.update로 expanded=True/state='running'을 유지해야 한다."""
    sid = "test-panel-running"
    SessionManager.reset_all_state(sid)

    build = initial_build_state("doc.pdf", "hash1")
    build.phase = BuildPhase.CHUNKING
    build.pct = 50
    SessionManager.set("build_state", build, session_id=sid)
    SessionManager.set("is_building_rag", True, session_id=sid)

    with _mock_st_status_container() as (mock_st, container):
        render_document_panel(sid)

    # st.status 진입 시점 인자 확인
    assert mock_st.status.call_args.kwargs["expanded"] is True
    assert mock_st.status.call_args.kwargs["state"] == "running"

    # __exit__ 자동 접힘을 무력화하는 명시적 update 확인
    container.update.assert_called_once()
    assert container.update.call_args.kwargs["expanded"] is True
    assert container.update.call_args.kwargs["state"] == "running"


def test_done_panel_explicitly_updates_expanded_false():
    """DONE 상태는 expanded=False/state='complete'로 고정되어야 한다."""
    sid = "test-panel-done"
    SessionManager.reset_all_state(sid)

    build = initial_build_state("doc.pdf", "hash1")
    complete_build(build, cache_used=True)
    SessionManager.set("build_state", build, session_id=sid)

    with _mock_st_status_container() as (mock_st, container):
        render_document_panel(sid)

    assert mock_st.status.call_args.kwargs["expanded"] is False
    assert mock_st.status.call_args.kwargs["state"] == "complete"

    container.update.assert_called_once()
    assert container.update.call_args.kwargs["expanded"] is False
    assert container.update.call_args.kwargs["state"] == "complete"


def test_error_panel_explicitly_updates_expanded_true():
    """ERROR 상태는 expanded=True/state='error'로 고정되어 재시도 CTA를 노출해야 한다."""
    sid = "test-panel-error"
    SessionManager.reset_all_state(sid)

    build = initial_build_state("doc.pdf", "hash1")
    fail_build(build, "parsing failed")
    SessionManager.set("build_state", build, session_id=sid)

    with _mock_st_status_container() as (mock_st, container):
        render_document_panel(sid)

    assert mock_st.status.call_args.kwargs["expanded"] is True
    assert mock_st.status.call_args.kwargs["state"] == "error"

    container.update.assert_called_once()
    assert container.update.call_args.kwargs["expanded"] is True
    assert container.update.call_args.kwargs["state"] == "error"
