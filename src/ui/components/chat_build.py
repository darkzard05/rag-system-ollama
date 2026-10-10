"""Document analysis (build) progress render component (shim).

Greenfield implementation lives in ``ui.components.doc_analysis``.
This module keeps the historical public names and delegates to the
single ``render_document_panel`` renderer. Flat legacy keys
(``rebuild_progress``/``rebuild_status``/``rebuild_done``) are never read
here (3B immediate deletion).
"""

from typing import Any

import streamlit as st

from core.session import SessionManager
from ui.components.common import AVATARS, ui_error  # noqa: F401
from ui.components.doc_analysis.actions import on_cancel as _on_cancel
from ui.components.doc_analysis.actions import on_retry as _on_retry
from ui.components.doc_analysis.panel import render_document_panel
from ui.strings import t
from ui.widget_keys import SAMPLE_QUESTION_STATE_KEY  # noqa: F401

__all__ = [
    "_cancel_rebuild",
    "_render_build_card",
    "_render_build_progress_block",
    "_render_build_progress_fragment",
    "_render_guidance_panel",
    "get_build_error_actions",
]


def _cancel_rebuild(sid: str) -> None:
    # on_click 콜백 종료 후 Streamlit 자동 post-callback 리런에 의존 (명시 rerun 금지).
    """문서 분석 재구축 취소 요청 콜백입니다."""
    _on_cancel(sid)


def _retry_build(sid: str) -> None:
    """문서 분석 실패/취소 시 파이프라인 재구축을 트리거하는 콜백."""
    _on_retry(sid)


def get_build_error_actions(error_msg: str) -> dict[str, Any]:
    """Return structured build error info with retry CTA.

    Called by test_error_recovery.py and render code.
    """
    return {
        "message": error_msg,
        "cause": error_msg,
        "retryable": True,
        "retry_kind": "rebuild",
        "cta": [t("action_retry_build")],
    }


def _render_build_progress_block(sid: str) -> None:
    """문서 분석 카드 렌더 (단일 패널 위임, BuildState 전용)."""
    render_document_panel(sid)


def _render_build_card(sid: str, build: Any = None) -> None:
    """하위 호환 별칭: 실제 렌더는 sid 기준으로 단일 패널에 위임한다."""
    _ = build
    render_document_panel(sid)


@st.fragment()
def _render_build_progress_fragment(sid: str) -> None:
    """빌드 상태 카드 전용 fragment (run_every 없음 — 게이트 테스트 고정).

    진행 중 갱신은 빌드 워커의 상태 전파 rerun + fragment 내 인터랙션
    (취소 버튼) rerun으로 전달된다. run_every 폴링은 빌드 완료 후 전체
    rerun 시 스테일 타이머 경고를 유발하므로 재도입하지 않는다
    (``test_build_progress_fragment_has_no_run_every``). D4: 대신 워커
    실패 경로의 에러 표면화(``async_worker._on_complete`` 상태 로그)로
    조용한 멈춤을 방지한다. 완료/취소/에러 상태에서도 카드는
    그대로 남아 기록으로 잔존한다.
    """
    _render_build_progress_block(sid)

    # on_click 콜백 종료 후 Streamlit 자동 post-callback 리런에 의존 (명시 rerun 금지).


def _on_sample_question_click(question: str) -> None:
    st.session_state[SAMPLE_QUESTION_STATE_KEY] = question


def _render_guidance_panel() -> None:
    """온보딩 단일 소스: 파일 미업로드 시에만 안내 카드를 렌더한다."""
    sid = SessionManager.get_session_id()
    if SessionManager.get("last_uploaded_file_name", "", sid):
        return

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        st.markdown(t("onboarding_guide"))
