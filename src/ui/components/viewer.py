"""
PDF 뷰어 및 문서 관련 UI 컴포넌트.
"""

import logging
import os

import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

from common.config import MSG_PDF_VIEWER_NO_FILE
from common.constants import UIConstants
from common.utils import safe_cache_data, safe_cache_resource
from core.session import SessionManager

logger = logging.getLogger(__name__)


@safe_cache_data(show_spinner=False)
def _get_pdf_bytes(pdf_path: str) -> bytes:
    if not os.path.exists(pdf_path):
        return b""
    try:
        with open(pdf_path, "rb") as f:
            return f.read()
    except Exception as e:
        logger.error(f"PDF 파일 읽기 실패 ({pdf_path}): {e}")
        return b""


@safe_cache_resource(show_spinner=False)
def _get_pdf_total_pages(pdf_path: str) -> int:
    import fitz

    if not os.path.exists(pdf_path):
        return 0
    try:
        with fitz.open(pdf_path) as doc:
            return len(doc)
    except Exception as e:
        logger.error(f"PDF 페이지 수 조회 실패: {e}")
        return 0


def render_pdf_column():
    pdf_path_raw = SessionManager.get("pdf_file_path")
    if not pdf_path_raw:
        st.info(MSG_PDF_VIEWER_NO_FILE)
        return

    pdf_path = os.path.abspath(pdf_path_raw)
    file_hash = SessionManager.get("file_hash", "none")

    total_pages = _get_pdf_total_pages(pdf_path)
    if total_pages == 0:
        st.error("⚠️ PDF 로드 실패: 파일이 손상되었거나 경로가 올바르지 않습니다.")
        return

    target_page = SessionManager.get("pdf_target_page")
    if target_page is not None:
        current_page = min(max(1, int(target_page)), total_pages)
        SessionManager.set("current_page", current_page)
        SessionManager.set("pdf_target_page", None)
        st.session_state["pdf_nav_input_v6"] = current_page
    else:
        if "pdf_nav_input_v6" in st.session_state:
            current_page = min(
                max(1, int(st.session_state["pdf_nav_input_v6"])), total_pages
            )
            SessionManager.set("current_page", current_page)
        else:
            current_page = min(
                max(1, SessionManager.get("current_page", 1)), total_pages
            )
            st.session_state["pdf_nav_input_v6"] = current_page

    # 하드코딩 제거 및 UIConstants 적용 (실제 높이는 CSS calc가 덮어씌움)
    with st.container(height=UIConstants.CONTAINER_HEIGHT, border=False):
        _display_pdf_viewer(pdf_path, current_page, file_hash)

    # [최적화] 네비게이션을 하단으로 이동하여 우측 채팅 입력창과 대칭 구조 형성
    _display_pdf_controls(current_page, total_pages)


def _display_pdf_viewer(pdf_path, current_page, file_hash):
    try:
        pdf_bytes = _get_pdf_bytes(pdf_path)
        if not pdf_bytes:
            st.error("⚠️ PDF 데이터를 불러올 수 없습니다.")
            return

        annotations = SessionManager.get("pdf_annotations", [])
        viewer_key = f"pdf_v7_{file_hash}_{current_page}_{len(annotations)}"

        pdf_viewer(
            input=pdf_bytes,
            render_text=True,
            pages_to_render=[current_page],
            annotations=annotations,
            annotation_outline_size=2,
            key=viewer_key,
        )
    except Exception as e:
        logger.error(f"PDF 뷰어 렌더링 오류: {e}", exc_info=True)
        st.error(f"PDF 뷰어 오류: {e}")


def _display_pdf_controls(current_page, total_pages):
    try:
        # [개선] 4컬럼 -> 3컬럼 구조로 변경하여 중앙 집중형 레이아웃 구현
        c_prev, c_center, c_next = st.columns(
            [1, 2, 1], gap="small", vertical_alignment="center"
        )

        with c_prev:
            if st.button(
                "⬅️ 이전",
                use_container_width=True,
                key="btn_nav_prev_v6",
                disabled=current_page <= 1,
            ):
                new_page = max(1, current_page - 1)
                SessionManager.set("pdf_target_page", new_page)
                SessionManager.set("current_page", new_page)
                st.rerun()

        with c_center:
            # "Page [X] of Y" 스타일을 위한 내부 정밀 레이아웃
            inner_col1, inner_col2, inner_col3 = st.columns(
                [0.7, 1, 1.3], gap="none", vertical_alignment="center"
            )

            with inner_col1:
                st.markdown(
                    "<div style='text-align: right; font-weight: 600; opacity: 0.8; padding-top: 2px;'>Page</div>",
                    unsafe_allow_html=True,
                )

            with inner_col2:

                def on_page_change():
                    new_p = st.session_state.get("pdf_nav_input_v6")
                    if new_p:
                        SessionManager.set("current_page", new_p)

                st.number_input(
                    "P",
                    min_value=1,
                    max_value=total_pages,
                    key="pdf_nav_input_v6",
                    on_change=on_page_change,
                    label_visibility="collapsed",
                )

            with inner_col3:
                st.markdown(
                    f"<div style='text-align: left; font-weight: 600; opacity: 0.8; padding-top: 2px;'>of {total_pages}</div>",
                    unsafe_allow_html=True,
                )

        with c_next:
            if st.button(
                "다음 ➡️",
                use_container_width=True,
                key="btn_nav_next_v6",
                disabled=current_page >= total_pages,
            ):
                new_page = min(total_pages, current_page + 1)
                SessionManager.set("pdf_target_page", new_page)
                SessionManager.set("current_page", new_page)
                st.rerun()

    except Exception as e:
        logger.error(f"PDF 컨트롤바 오류: {e}", exc_info=True)
