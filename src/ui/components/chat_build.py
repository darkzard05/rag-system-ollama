"""Document analysis (build) progress render component (slim shim)."""

from typing import Any

import streamlit as st

from ui.components.common import AVATARS, ui_error  # noqa: F401
from ui.components.doc_analysis.actions import on_cancel as _cancel_rebuild
from ui.components.doc_analysis.panel import render_document_panel
from ui.strings import t
from ui.widget_keys import SAMPLE_QUESTION_STATE_KEY

__all__ = [
    "_cancel_rebuild",
    "_render_build_card",
    "_render_build_progress_block",
    "_render_build_progress_fragment",
    "_render_guidance_panel",
    "get_build_error_actions",
]


def _render_build_progress_block(sid: str) -> None:
    render_document_panel(sid)


def _render_build_card(sid: str, build: Any = None) -> None:
    _ = build
    render_document_panel(sid)


def get_build_error_actions(error_msg: str) -> dict[str, Any]:
    return {
        "message": error_msg,
        "cause": error_msg,
        "retryable": True,
        "retry_kind": "rebuild",
        "cta": [t("action_retry_build")],
    }


@st.fragment()
def _render_build_progress_fragment(sid: str) -> None:
    render_document_panel(sid)


def _on_sample_question_click(question: str) -> None:
    st.session_state[SAMPLE_QUESTION_STATE_KEY] = question


def _render_guidance_panel() -> None:
    from core.session import SessionManager

    sid = SessionManager.get_session_id()
    if SessionManager.get("last_uploaded_file_name", "", sid):
        return

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        st.markdown(t("onboarding_guide"))
        st.caption(t("focus_order_notice"))
