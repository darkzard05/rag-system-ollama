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


@st.fragment
def _pdf_viewer_fragment():
    from streamlit_pdf_viewer import pdf_viewer

    viewer_container = st.container()

    pdf_path_raw = SessionManager.get("pdf_file_path", None)
    if not pdf_path_raw:
        with viewer_container:
            st.info(MSG_PDF_VIEWER_NO_FILE)
        return

    pdf_path = os.path.abspath(pdf_path_raw)
    try:
        # [최적화] 바이트 데이터와 페이지 수를 각각 분리하여 캐싱 (Context7 권장)
        total_pages = _get_pdf_total_pages(pdf_path)
        pdf_bytes = _get_pdf_bytes(pdf_path)

        if total_pages == 0:
            with viewer_container:
                st.error("⚠️ PDF 로드 실패: 페이지를 찾을 수 없습니다.")
            return

        # 1. 점프 타겟 확인 및 현재 페이지 설정
        target_page = SessionManager.get("pdf_target_page")

        if target_page:
            st.session_state.current_page = target_page
            SessionManager.set("current_page", target_page)
            SessionManager.set("pdf_target_page", None)  # 처리 후 초기화

        if "current_page" not in st.session_state:
            st.session_state.current_page = SessionManager.get("current_page", 1)

        current_page = min(max(1, st.session_state.current_page), total_pages)

        # --- [추가] 문서 처리 상태 확인 (비활성화용) ---
        pdf_processed = SessionManager.get("pdf_processed", False)
        pdf_error = SessionManager.get("pdf_processing_error")
        is_processing = bool(pdf_path_raw and not pdf_processed and not pdf_error)

        # [수정] 전역 변수가 아닌 활성 메시지에서 하이라이트(Annotations) 가져오기
        active_idx = st.session_state.get("active_msg_index")
        messages = SessionManager.get_messages()

        annotations = []
        if active_idx is not None and active_idx < len(messages):
            msg = messages[active_idx]
            annotations = msg.get("annotations", [])
        else:
            annotations = SessionManager.get("pdf_annotations", [])

        # [수정] 데이터 변경 시점을 key에 반영하여 강제 갱신 유도
        anno_key_part = f"{len(annotations)}_{id(annotations)}"

        with viewer_container:
            # 뷰어 렌더링 (높이 650 고정)
            # [최적화] key에 target_page와 anno_key_part를 포함시켜 프래그먼트 내부 즉시 갱신
            pdf_viewer(
                input=pdf_bytes,
                render_text=True,
                pages_to_render=[current_page],
                annotations=annotations,
                annotation_outline_size=2,
                height=650,
                key=f"pdf_viewer_p{current_page}_{anno_key_part}_{hash(pdf_path)}",
            )

        # --- 하단 컨트롤바 ---
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        c_prev, c_input, c_next = st.columns([1, 3, 1])

        with c_prev:
            if st.button(
                "⬅️",
                use_container_width=True,
                disabled=is_processing or current_page <= 1,
            ):
                st.session_state.current_page -= 1
                SessionManager.set("current_page", st.session_state.current_page)
                st.rerun()

        with c_input:

            def on_page_change():
                new_p = st.session_state.page_nav_input
                st.session_state.current_page = new_p
                SessionManager.set("current_page", new_p)

            st.number_input(
                f"Page (of {total_pages})",
                min_value=1,
                max_value=total_pages,
                value=current_page,
                key="page_nav_input",
                on_change=on_page_change,
                label_visibility="collapsed",
                disabled=is_processing,
            )

        with c_next:
            if st.button(
                "➡️",
                use_container_width=True,
                disabled=is_processing or current_page >= total_pages,
            ):
                st.session_state.current_page += 1
                SessionManager.set("current_page", st.session_state.current_page)
                st.rerun()

    except Exception as e:
        logger.error(f"PDF 뷰어 렌더링 오류: {e}", exc_info=True)
        st.error(f"PDF 오류: {e}")


def render_pdf_viewer():
    """PDF 뷰어 최상위 렌더링 함수"""
    _pdf_viewer_fragment()
