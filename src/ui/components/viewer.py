"""
PDF 뷰어 및 문서 관련 UI 컴포넌트.
@st.fragment를 사용하여 PDF 렌더링과 네비게이션이 전체 페이지 리런 없이 독립적으로 업데이트됩니다.
"""

import logging
import os

import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

from common.config import MSG_PDF_VIEWER_NO_FILE
from common.exceptions import PDFProcessingError
from common.utils import safe_cache_data
from core.session import SessionManager
from ui.widget_keys import pdf_viewer_key

logger = logging.getLogger(__name__)


@safe_cache_data(ttl=60, show_spinner=False)
def _get_pdf_bytes(pdf_path: str) -> bytes:
    if not os.path.exists(pdf_path):
        return b""
    try:
        with open(pdf_path, "rb") as f:
            return f.read()
    except OSError as e:
        logger.error(f"PDF 파일 읽기 실패 ({pdf_path}): {e}")
        return b""
    except PDFProcessingError as e:
        logger.error(f"PDF 처리 실패 ({pdf_path}): {e}")
        return b""


@safe_cache_data(ttl=300, show_spinner=False)
def _get_pdf_total_pages(pdf_path: str) -> int:
    import fitz

    if not os.path.exists(pdf_path):
        return 0
    try:
        with fitz.open(pdf_path) as doc:
            return len(doc)
    except (RuntimeError, ValueError) as e:
        logger.error(f"PDF 페이지 수 조회 실패: {e}")
        raise PDFProcessingError(
            "PDF 페이지 수 조회 실패", details={"error": str(e)}
        ) from e


# ---------------------------------------------------------------------------
# Navigation callbacks (module-level to avoid per-rerun recreation)
# ---------------------------------------------------------------------------


def _on_prev_click():
    """이전 페이지로 이동 (네비게이션 버튼 콜백)"""
    current = SessionManager.get("current_page", 1)
    new_page = max(1, current - 1)
    SessionManager.set("current_page", new_page)
    st.session_state["pdf_nav_input_v6"] = new_page


def _on_next_click(total_pages: int):
    """다음 페이지로 이동 (네비게이션 버튼 콜백)"""
    current = SessionManager.get("current_page", 1)
    new_page = min(total_pages, current + 1)
    SessionManager.set("current_page", new_page)
    st.session_state["pdf_nav_input_v6"] = new_page


def _on_page_change():
    """페이지 번호 입력 변경 시 (number_input on_change 콜백)"""
    new_p = st.session_state.get("pdf_nav_input_v6")
    if new_p:
        SessionManager.set("current_page", new_p)


def _on_next_click_callback():
    """다음 페이지 네비게이션 콜백 (module-level)"""
    pdf_path = SessionManager.get("pdf_file_path", "")
    if pdf_path:
        total = _get_pdf_total_pages(os.path.abspath(pdf_path))
        _on_next_click(total)


# ---------------------------------------------------------------------------
# PDF state resolution (shared by viewer and controls fragments)
# ---------------------------------------------------------------------------


def _resolve_pdf_state() -> dict | None:
    """Resolve current PDF state from session.

    Returns dict with keys: pdf_path, file_hash, total_pages, current_page.
    Returns None if no PDF is loaded or PDF is invalid.
    Handles external page navigation (e.g., from chat references via pdf_target_page).
    """
    pdf_path_raw = SessionManager.get("pdf_file_path")
    if not pdf_path_raw:
        return None

    pdf_path = os.path.abspath(pdf_path_raw)
    file_hash = SessionManager.get("file_hash", "none")

    total_pages = _get_pdf_total_pages(pdf_path)
    if total_pages == 0:
        return None

    # Handle external page navigation (e.g., from chat references)
    target_page = SessionManager.get("pdf_target_page")
    if target_page is not None:
        current_page = min(max(1, int(target_page)), total_pages)
        SessionManager.set("current_page", current_page)
        # pdf_target_page는 일회성 소비: 점프 적용 후 키를 삭제하여
        # 사용자가 수동 네비게이션으로 벗어나도 매 rerun마다 참조 페이지로
        # 되돌아가지 않도록 보장한다.
        SessionManager.delete("pdf_target_page")
        st.session_state.pop("pdf_target_page", None)
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

    return {
        "pdf_path": pdf_path,
        "file_hash": file_hash,
        "total_pages": total_pages,
        "current_page": current_page,
    }


# ---------------------------------------------------------------------------
# Fragment: PDF 뷰어 (독립 스크롤 영역 — 네비게이션과 분리)
# ---------------------------------------------------------------------------


@st.fragment
def render_pdf_viewer():
    """
    PDF 뷰어를 렌더링하는 fragment (컨트롤 없음).

    네비게이션 컨트롤과 분리되어 독립적으로 업데이트됩니다.
    fragment 내부의 위젯 액션은 이 fragment만 재실행하여
    전체 페이지 리런 없이 PDF를 업데이트합니다.
    """
    state = _resolve_pdf_state()
    if not state:
        st.info(MSG_PDF_VIEWER_NO_FILE)
        return

    _display_pdf_viewer(state["pdf_path"], state["current_page"], state["file_hash"])


# ---------------------------------------------------------------------------
# Fragment: PDF 네비게이션 컨트롤 (하단 컨트롤 바 — 뷰어와 분리)
# ---------------------------------------------------------------------------


@st.fragment
def render_pdf_controls():
    """
    PDF 네비게이션 컨트롤을 렌더링하는 fragment (뷰어 없음).

    PDF 열 하단(뷰어 아래)에 위치하며, 뷰어 fragment와 분리되어
    독립적으로 업데이트됩니다. 버튼 클릭 시 이 fragment만 재실행되어
    전체 페이지 리런을 방지합니다.
    """
    state = _resolve_pdf_state()
    if not state:
        return

    _display_pdf_controls(state["current_page"], state["total_pages"])


def _display_pdf_viewer(pdf_path, current_page, file_hash):
    try:
        pdf_bytes = _get_pdf_bytes(pdf_path)
        if not pdf_bytes:
            st.error("⚠️ PDF 데이터를 불러올 수 없습니다.")
            return

        annotations = SessionManager.get("pdf_annotations", [])
        viewer_key = pdf_viewer_key(file_hash)

        pdf_viewer(
            input=pdf_bytes,
            render_text=True,
            pages_to_render=[current_page],
            annotations=annotations,
            annotation_outline_size=2,
            key=viewer_key,
        )
    except PDFProcessingError as e:
        logger.error(f"PDF 처리 오류: {e}")
        st.error(f"PDF 뷰어 오류: {e}")
    except Exception as e:
        logger.error(f"PDF 뷰어 렌더링 오류: {e}", exc_info=True)
        st.error(f"PDF 뷰어 렌더링 오류: {e}")


def _display_pdf_controls(current_page, total_pages):
    try:
        col_prev, col_page, col_next = st.columns(
            [1, 2, 1], gap="small", vertical_alignment="center"
        )

        with col_prev:
            st.button(
                "⬅️ 이전",
                use_container_width=True,
                key="btn_nav_prev_v6",
                disabled=current_page <= 1,
                on_click=_on_prev_click,
            )

        with col_page:
            st.number_input(
                "Page",
                min_value=1,
                max_value=total_pages,
                key="pdf_nav_input_v6",
                on_change=_on_page_change,
                label_visibility="collapsed",
            )

        with col_next:
            st.button(
                "다음 ➡️",
                use_container_width=True,
                key="btn_nav_next_v6",
                disabled=current_page >= total_pages,
                on_click=_on_next_click_callback,
            )

    except Exception as e:
        logger.error(f"PDF 컨트롤바 오류: {e}", exc_info=True)
