"""
PDF 뷰어 및 문서 관련 UI 컴포넌트.
"""

import logging
import os

import streamlit as st

from common.config import MSG_PDF_VIEWER_NO_FILE
from common.utils import safe_cache_resource
from core.session import SessionManager

logger = logging.getLogger(__name__)


@safe_cache_resource(show_spinner=False)
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

    # [수정] 높이 제약 제거
    viewer_container = st.container()

    pdf_path_raw = SessionManager.get("pdf_file_path", None)
    if not pdf_path_raw:
        with viewer_container:
            st.info(MSG_PDF_VIEWER_NO_FILE)
        return

    pdf_path = os.path.abspath(pdf_path_raw)
    try:
        total_pages, pdf_bytes = _get_pdf_info(pdf_path)

        # [디버그] 뷰어 프래그먼트 진입 로그
        logger.info(f"[DEBUG] PDF 뷰어 로드 시작 (전체 {total_pages}p)")

        if total_pages == 0:
            with viewer_container:
                st.error("⚠️ PDF 로드 실패: 페이지를 찾을 수 없습니다.")
            return

        # 1. 점프 타겟 확인 및 현재 페이지 설정
        if "current_page" not in st.session_state:
            st.session_state.current_page = SessionManager.get("current_page", 1)

        # [수정] SessionManager에서 점프 타겟 확인 (st.session_state보다 신뢰성 높음)
        target_page = SessionManager.get("pdf_target_page")
        logger.info(f"[DEBUG] 점프 타겟 확인: {target_page}")

        if target_page:
            logger.info(
                f"[DEBUG] 페이지 이동 실행: {st.session_state.current_page} -> {target_page}"
            )
            st.session_state.current_page = target_page
            SessionManager.set("current_page", target_page)
            SessionManager.set("pdf_target_page", None)  # 처리 후 초기화

        current_page = min(max(1, st.session_state.current_page), total_pages)
        logger.info(f"[DEBUG] 최종 렌더링 페이지: {current_page}")

        # --- [추가] 문서 처리 상태 확인 (비활성화용) ---
        pdf_processed = SessionManager.get("pdf_processed", False)
        pdf_error = SessionManager.get("pdf_processing_error")
        is_processing = bool(pdf_path_raw and not pdf_processed and not pdf_error)

        # [추가] 하이라이트 좌표(Annotations) 가져오기
        annotations = SessionManager.get("pdf_annotations", [])

        # [디버그] 현재 페이지에 해당하는 하이라이트 개수 확인
        current_page_annos = [a for a in annotations if a.get("page") == current_page]

        # [수정] 데이터 변경 시점을 key에 반영하여 강제 갱신 유도
        # annotations의 메모리 주소나 길이를 활용
        anno_key_part = f"{len(annotations)}_{id(annotations)}"

        logger.info(
            f"[DEBUG] Viewer Page {current_page}: Rendering {len(current_page_annos)} annotations (Total: {len(annotations)})"
        )

        with viewer_container:
            # 뷰어 렌더링 (높이 650 고정)
            pdf_viewer(
                input=pdf_bytes,
                render_text=True,
                pages_to_render=[current_page],
                annotations=annotations,
                annotation_outline_size=2,
                height=650,
                key=f"pdf_viewer_p{current_page}_{anno_key_part}_{hash(pdf_path_raw)}",
            )

        # --- 하단 컨트롤바 ---
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        c_prev, c_input, c_next = st.columns([1, 3, 1])

        with c_prev:
            # [수정] 처리 중이거나 1페이지면 비활성화
            if st.button(
                "⬅️",
                use_container_width=True,
                disabled=is_processing or current_page <= 1,
            ):
                st.session_state.current_page -= 1
                SessionManager.set("current_page", st.session_state.current_page)
                st.rerun()

        with c_input:
            # [수정] 위젯의 value가 항상 계산된 current_page를 따르도록 강제
            def on_page_change():
                new_p = st.session_state.page_nav_input
                SessionManager.set("current_page", new_p)
                # 즉시 상태 업데이트를 위해 세션 매니저와 동기화

            st.number_input(
                f"Page (of {total_pages})",
                min_value=1,
                max_value=total_pages,
                value=current_page,  # 이 값이 이제 점프 시 바뀐 값으로 들어감
                key="page_nav_input",
                on_change=on_page_change,
                label_visibility="collapsed",
                disabled=is_processing,
            )

        with c_next:
            # [수정] 처리 중이거나 마지막 페이지면 비활성화
            if st.button(
                "➡️",
                use_container_width=True,
                disabled=is_processing or current_page >= total_pages,
            ):
                st.session_state.current_page += 1
                SessionManager.set("current_page", st.session_state.current_page)
                st.rerun()

    except Exception as e:
        st.error(f"PDF 오류: {e}")


def render_pdf_viewer():
    """PDF 뷰어 최상위 렌더링 함수"""
    _pdf_viewer_fragment()
