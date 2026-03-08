"""
PDF 뷰어 및 문서 관련 UI 컴포넌트.
"""

import logging
import os

import streamlit as st

from common.config import MSG_PDF_VIEWER_NO_FILE
from common.utils import safe_cache_data, safe_cache_resource
from core.session import SessionManager

logger = logging.getLogger(__name__)


@safe_cache_data(show_spinner=False)
def _get_pdf_bytes(pdf_path: str) -> bytes:
    """PDF 파일의 바이트 데이터를 캐싱하여 로드합니다."""
    if not os.path.exists(pdf_path):
        return b""
    with open(pdf_path, "rb") as f:
        return f.read()


@safe_cache_resource(show_spinner=False)
def _get_pdf_total_pages(pdf_path: str) -> int:
    """PDF의 총 페이지 수를 캐싱합니다."""
    import fitz

    if not os.path.exists(pdf_path):
        return 0
    try:
        with fitz.open(pdf_path) as doc:
            return len(doc)
    except Exception as e:
        logger.error(f"PDF 페이지 수 조회 실패: {e}")
        return 0


@st.fragment()
def _pdf_viewer_fragment():
    """
    PDF 뷰어 프래그먼트 (격리된 리런 영역)
    [최적화] st.rerun(scope="fragment")를 사용하여 UI의 독립적 갱신을 보장합니다.
    """
    from streamlit_pdf_viewer import pdf_viewer

    # 1. 상태 동기화 및 타겟 페이지 처리
    target_page = SessionManager.get("pdf_target_page")
    if target_page is not None:
        st.session_state.current_page = int(target_page)
        SessionManager.set("current_page", int(target_page))
        SessionManager.set("pdf_target_page", None)

    if "current_page" not in st.session_state:
        st.session_state.current_page = SessionManager.get("current_page", 1)

    pdf_path_raw = SessionManager.get("pdf_file_path")
    if not pdf_path_raw:
        st.info(MSG_PDF_VIEWER_NO_FILE)
        return

    pdf_path = os.path.abspath(pdf_path_raw)
    try:
        total_pages = _get_pdf_total_pages(pdf_path)
        pdf_bytes = _get_pdf_bytes(pdf_path)

        if total_pages == 0:
            st.error("⚠️ PDF 로드 실패")
            return

        current_page = min(max(1, st.session_state.current_page), total_pages)

        # 하이라이트 어노테이션
        active_idx = st.session_state.get("active_msg_index")
        messages = SessionManager.get_messages() or []
        annotations = []
        if active_idx is not None and active_idx < len(messages):
            annotations = messages[active_idx].get("annotations", [])
        if not annotations:
            annotations = SessionManager.get("pdf_annotations", [])

        viewer_key = f"pdf_v5_{hash(pdf_path)}_{current_page}_{len(annotations)}"

        # 3. PDF 뷰어 렌더링
        pdf_viewer(
            input=pdf_bytes,
            render_text=True,
            pages_to_render=[current_page],
            annotations=annotations,
            annotation_outline_size=2,
            height=650,
            key=viewer_key,
        )

        # 4. 하단 컨트롤바
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        c_prev, c_input, c_next = st.columns([1, 2, 1])

        with c_prev:
            if st.button(
                "⬅️",
                use_container_width=True,
                key="btn_nav_prev",
                disabled=current_page <= 1,
            ):
                st.session_state.current_page -= 1
                SessionManager.set("current_page", st.session_state.current_page)
                st.rerun(scope="fragment")

        with c_input:

            def on_page_change():
                new_p = st.session_state.get("page_nav_input_v5", current_page)
                st.session_state.current_page = new_p
                SessionManager.set("current_page", new_p)

            st.number_input(
                f"Page / {total_pages}",
                min_value=1,
                max_value=total_pages,
                value=current_page,
                key="page_nav_input_v5",
                on_change=on_page_change,
                label_visibility="collapsed",
            )

        with c_next:
            if st.button(
                "➡️",
                use_container_width=True,
                key="btn_nav_next",
                disabled=current_page >= total_pages,
            ):
                st.session_state.current_page += 1
                SessionManager.set("current_page", st.session_state.current_page)
                st.rerun(scope="fragment")

    except Exception as e:
        logger.error(f"PDF 뷰어 오류: {e}", exc_info=True)
        st.error(f"PDF 오류: {e}")


def render_pdf_viewer():
    """PDF 뷰어 최상위 렌더링 함수"""
    _pdf_viewer_fragment()
