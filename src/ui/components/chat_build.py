"""문서 분석 (빌드) 진행 렌더 컴포넌트.

``chat.py`` (P3 분리)에서 이동한 빌드 진행/문서 컨텍스트 렌더 함수들:
- ``_cancel_rebuild`` — 분석 재구축 취소 요청 콜백
- ``_render_build_progress_block`` — 빌드 상태 블록 단독 렌더
- ``_render_build_progress_fragment`` — 1.5초 폴링 빌드 상태 fragment
- ``_render_guidance_panel`` — 빈 대화 가이드 패널
- ``_render_doc_context_inline`` — 타임라인 첫 메시지 문서 컨텍스트

본 모듈은 ``chat.py``를 import하지 않는다 (순환 의존 방지).
"""

import streamlit as st

from common.config import MSG_CHAT_GUIDE
from core.session import SessionManager
from ui.components.common import AVATARS, status_line
from ui.widget_keys import cancel_rebuild_key

__all__ = [
    "_cancel_rebuild",
    "_render_build_progress_block",
    "_render_build_progress_fragment",
    "_render_doc_context_inline",
    "_render_guidance_panel",
]


def _cancel_rebuild(sid: str) -> None:
    """문서 분석 재구축 취소 요청 콜백입니다."""
    SessionManager.set("rebuild_cancelled", True, session_id=sid)
    st.rerun()


def _render_build_progress_block(sid: str) -> None:
    """문서 분석 상태 블록을 단독 렌더합니다 (전용 폴링 fragment가 호출).

    리팩터링에서 타임라인 폴링이 제거된 뒤, 빌드 도중에는 전체 rerun이
    발생하지 않아 ``st.progress`` 가 0%에 고착되던 결함(진행 바 동결)을
    해결하기 위해 분리했다. ``_report_progress``(main.py)가 갱신하는
    ``rebuild_progress`` 상태만 읽어 주기적(``run_every``)으로 다시 그린다.

    분석 블록은 대화의 일부로 영구 잔존한다(빌드 완료/취소/에러 후에도 남아
    타임라인 기록으로 남는다). 빌드가 한 번도 시작되지 않은 초기 상태에서만
    렌더하지 않는다.
    """
    is_building = bool(SessionManager.get("is_building_rag", False, sid))
    is_cancelling = bool(SessionManager.get("rebuild_cancelled", False, sid))
    is_done = bool(SessionManager.get("rebuild_done", False, sid))
    has_doc = bool(SessionManager.get("last_uploaded_file_name", "", sid))

    # 빌드가 한 번도 시작되지 않은 초기 상태(업로드 전)에서는 노출하지 않는다.
    if not (is_building or is_cancelling or is_done or has_doc):
        return

    progress = int(SessionManager.get("rebuild_progress", 0, sid))
    status_text = str(SessionManager.get("rebuild_status", "", sid) or "")
    error = SessionManager.get("pdf_processing_error", "", sid) or ""

    if error:
        label, state, expanded = "Analysis failed/cancelled", "error", True
    elif progress >= 100 or is_done:
        label, state, expanded = "Analysis complete", "complete", False
    elif is_cancelling:
        label, state, expanded = "Cancelling analysis...", "running", True
    else:
        label, state, expanded = (
            f"Analyzing document: {status_text}",
            "running",
            True,
        )

    with (
        st.chat_message("system", avatar=AVATARS["building"]),
        st.status(label, expanded=expanded, state=state),
    ):
        st.progress(progress / 100)
        st.caption(f"{progress}% complete")
        if state == "running" and not is_cancelling:
            st.button(
                "Cancel Analysis",
                key=cancel_rebuild_key(sid),
                on_click=_cancel_rebuild,
                args=(sid,),
                use_container_width=True,
            )


@st.fragment(run_every=1.5)
def _render_build_progress_fragment(sid: str) -> None:
    """빌드 상태 블록 전용 폴링 fragment.

    전체 rerun 없이 1.5초마다 ``rebuild_progress`` 를 다시 읽어 진행 바를
    갱신한다(타임라인 폴링이 제거된 빈틈을 메움). 빌드 완료 후에는
    ``run_in_background_worker._on_complete`` 의 rerun이 최종 100%를 확정한다.
    완료/취소/에러 상태에서도 블록은 그대로 남아 대화 기록으로 잔존한다.
    """
    _render_build_progress_block(sid)


def _render_guidance_panel() -> None:
    """빈 대화 상태의 단일 가이드 메시지를 렌더링합니다."""
    st.chat_message("system").markdown(MSG_CHAT_GUIDE)


def _render_doc_context_inline(sid: str) -> None:
    """문서 컨텍스트를 타임라인 첫 메시지로 렌더링합니다 (네이티브)."""
    file_name = str(SessionManager.get("last_uploaded_file_name", "", sid) or "")
    if not file_name:
        pdf_path = str(SessionManager.get("pdf_file_path", "", sid) or "")
        if pdf_path:
            file_name = pdf_path.replace("\\", "/").rsplit("/", 1)[-1]
    if not file_name:
        return

    is_building = bool(SessionManager.get("is_building_rag", False, sid))
    is_ready = SessionManager.is_ready_for_chat(session_id=sid)
    has_error = bool(SessionManager.get("pdf_processing_error", "", sid))

    doc_stats = SessionManager.get("doc_stats", {}, sid) or {}
    doc_loaded = bool(SessionManager.get("pdf_processed", False, sid))
    cache_tag = ""
    if doc_loaded and doc_stats:
        cache_tag = " [cached]" if doc_stats.get("cache_used") else " [new]"

    with st.chat_message("system", avatar=AVATARS["document"]):
        if is_building:
            st.caption(status_line(file_name, f"Analyzing...{cache_tag}"))
            # 진행 상황은 메시지 루프에서 build_progress 타입으로 처리
        elif has_error:
            st.caption(status_line(file_name, f"Error{cache_tag}"))
        elif is_ready:
            st.caption(status_line(file_name, f"Ready{cache_tag}"))
        else:
            st.caption(status_line(file_name, f"Waiting...{cache_tag}"))
