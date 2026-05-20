"""
PDF 뷰어 및 문서 관련 UI 컴포넌트.
(Architectural Refactor: 통합 프래그먼트 구조로 동기화 문제 해결 및 독립 스크롤 적용)
"""

import logging
import os

import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

from common.config import MSG_PDF_VIEWER_NO_FILE
from common.utils import safe_cache_data, safe_cache_resource
from core.session import SessionManager

logger = logging.getLogger(__name__)


@safe_cache_data(show_spinner=False)
def _get_pdf_bytes(pdf_path: str) -> bytes:
    """PDF 파일의 바이트 데이터를 캐싱하여 로드합니다."""
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
def render_pdf_column():
    """
    PDF 뷰어 컬럼 통합 렌더링 함수.
    표시(Display)와 제어(Controls)를 하나의 프래그먼트로 묶어 완벽한 동기화를 보장합니다.
    App-Shell 구조를 위해 고정 높이 컨테이너 내에서 스크롤되도록 설계되었습니다.
    """
    pdf_path_raw = SessionManager.get("pdf_file_path")
    if not pdf_path_raw:
        st.info(MSG_PDF_VIEWER_NO_FILE)
        return

    pdf_path = os.path.abspath(pdf_path_raw)
    file_hash = SessionManager.get("file_hash", "none")

    # 1. 페이지 상태 결정 (Single Source of Truth)
    total_pages = _get_pdf_total_pages(pdf_path)
    if total_pages == 0:
        st.error("⚠️ PDF 로드 실패: 파일이 손상되었거나 경로가 올바르지 않습니다.")
        return

    # 외부(채팅 레퍼런스 등)에서의 페이지 이동 요청 처리
    target_page = SessionManager.get("pdf_target_page")
    if target_page is not None:
        current_page = min(max(1, int(target_page)), total_pages)
        SessionManager.set("current_page", current_page)
        SessionManager.set("pdf_target_page", None)
    else:
        current_page = min(max(1, SessionManager.get("current_page", 1)), total_pages)

    # 2. 상단 컨트롤 영역 (고정)
    _display_pdf_controls(current_page, total_pages)

    # 3. PDF 표시 영역 (독립 스크롤 컨테이너)
    with st.container(
        height=500, border=False
    ):  # CSS가 100%로 덮어씌울 것이므로 상징적인 값 사용
        _display_pdf_viewer(pdf_path, current_page, file_hash)


def _display_pdf_viewer(pdf_path, current_page, file_hash):
    """실제 PDF 렌더링 영역"""
    try:
        pdf_bytes = _get_pdf_bytes(pdf_path)
        if not pdf_bytes:
            st.error("⚠️ PDF 데이터를 불러올 수 없습니다.")
            return

        # 하이라이트 어노테이션 추출
        active_idx = st.session_state.get("active_msg_index")
        messages = SessionManager.get_messages() or []
        annotations = []
        if active_idx is not None and active_idx < len(messages):
            annotations = messages[active_idx].get("annotations", [])
        if not annotations:
            annotations = SessionManager.get("pdf_annotations", [])

        # 키 생성: 페이지 변경 및 어노테이션 개수 변화 감지 (리렌더링 트리거)
        viewer_key = f"pdf_v7_{file_hash}_{current_page}_{len(annotations)}"

        pdf_viewer(
            input=pdf_bytes,
            render_text=True,
            pages_to_render=[current_page],
            annotations=annotations,
            annotation_outline_size=2,
            height=None,
            key=viewer_key,
        )
    except Exception as e:
        logger.error(f"PDF 뷰어 렌더링 오류: {e}", exc_info=True)
        st.error(f"PDF 뷰어 오류: {e}")


def _display_pdf_controls(current_page, total_pages):
    """페이지 이동 컨트롤 영역"""
    try:
        c_prev, c_input, c_next = st.columns([1, 2, 1], gap="small")

        with c_prev:
            if st.button(
                "⬅️",
                use_container_width=True,
                key="btn_nav_prev_v6",
                disabled=current_page <= 1,
            ):
                SessionManager.set("current_page", max(1, current_page - 1))
                st.rerun(scope="fragment")

        with c_input:

            def on_page_change():
                new_p = st.session_state.get("pdf_nav_input_v6")
                if new_p:
                    SessionManager.set("current_page", new_p)

            st.number_input(
                f"Page / {total_pages}",
                min_value=1,
                max_value=total_pages,
                value=current_page,
                key="pdf_nav_input_v6",
                on_change=on_page_change,
                label_visibility="collapsed",
            )

        with c_next:
            if st.button(
                "➡️",
                use_container_width=True,
                key="btn_nav_next_v6",
                disabled=current_page >= total_pages,
            ):
                SessionManager.set("current_page", min(total_pages, current_page + 1))
                st.rerun(scope="fragment")

    except Exception as e:
        logger.error(f"PDF 컨트롤바 오류: {e}", exc_info=True)
