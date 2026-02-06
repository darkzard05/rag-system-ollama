"""
PDF 뷰어 및 문서 관련 UI 컴포넌트.
"""

import os

import streamlit as st

from common.config import MSG_PDF_VIEWER_NO_FILE
from core.session import SessionManager


@st.cache_resource(show_spinner=False)
def _get_pdf_info(pdf_path: str) -> tuple[int, bytes]:
    import fitz

    if not os.path.exists(pdf_path):
        return 0, b""
    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    return total_pages, pdf_bytes


@st.fragment
def _pdf_viewer_fragment():
    from streamlit_pdf_viewer import pdf_viewer

    # [최적화] 고정된 Viewport Height 기반의 높이 설정 (JS 의존성 감소)
    win_h = st.session_state.get("last_valid_height", 800)
    viewer_h = max(400, win_h - 250)

    viewer_container = st.container(height=viewer_h, border=True)

    pdf_path_raw = SessionManager.get("pdf_file_path", None)
    if not pdf_path_raw:
        with viewer_container:
            st.info(MSG_PDF_VIEWER_NO_FILE)
        return

    pdf_path = os.path.abspath(pdf_path_raw)
    try:
        total_pages, pdf_bytes = _get_pdf_info(pdf_path)
        if total_pages == 0:
            with viewer_container:
                st.error("⚠️ PDF 로드 실패")
            return

        if "pdf_page_index" not in st.session_state:
            st.session_state.pdf_page_index = SessionManager.get("current_page", 1)

        if "pdf_render_text" not in st.session_state:
            st.session_state.pdf_render_text = True

        viewer_params = {
            "input": pdf_bytes,
            "pages_to_render": [st.session_state.pdf_page_index],
            "render_text": st.session_state.get("pdf_render_text", True),
            "annotation_outline_size": 2,
        }

        with viewer_container:
            pdf_viewer(**viewer_params)

        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
        # 컨트롤바 우측 정렬
        c_spacer, c_page, c_set = st.columns([3.0, 1.5, 0.5])

        with c_page:
            sub_col1, sub_col2 = st.columns([1, 1])
            with sub_col1:
                safe_page_idx = min(
                    max(1, st.session_state.pdf_page_index), total_pages
                )
                st.number_input(
                    "Page",
                    min_value=1,
                    max_value=total_pages,
                    key="pdf_page_index_input",
                    value=safe_page_idx,
                    on_change=lambda: SessionManager.set(
                        "current_page", st.session_state.pdf_page_index_input
                    ),
                    label_visibility="collapsed",
                )
                st.session_state.pdf_page_index = st.session_state.pdf_page_index_input
            with sub_col2:
                st.markdown(
                    f"<div style='line-height: 2.3rem; white-space: nowrap;'>/ {total_pages} p</div>",
                    unsafe_allow_html=True,
                )

        with c_set, st.popover("⚙️", use_container_width=True):
            st.caption("📝 텍스트 설정")
            st.session_state.pdf_render_text = st.toggle(
                "텍스트 선택 가능", value=st.session_state.pdf_render_text
            )

    except Exception as e:
        st.error(f"PDF 오류: {e}")


def render_pdf_viewer():
    """PDF 뷰어 최상위 렌더링 함수"""
    _pdf_viewer_fragment()
