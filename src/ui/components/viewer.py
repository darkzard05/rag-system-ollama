"""
PDF 뷰어 및 문서 관련 UI 컴포넌트.
@st.fragment를 사용하여 PDF 렌더링과 네비게이션이 전체 페이지 리런 없이 독립적으로 업데이트됩니다.
"""

import logging
import os
import time

import streamlit as st

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
def _get_pdf_total_pages(pdf_path: str) -> int | None:
    """PDF 총 페이지 수 반환. 열 수 없으면 None (절대 raise하지 않음).

    st.cache_data 캐시 계층은 캐시 함수가 raise한 예외를 재전파하므로
    손상된 PDF가 스크립트를 죽이지 않도록 반드시 여기서 소화한다.
    """
    import fitz

    if not os.path.exists(pdf_path):
        return None
    try:
        with fitz.open(pdf_path) as doc:
            return len(doc)
    except PDFProcessingError as e:
        logger.error(f"PDF 페이지 수 조회 실패: {e}")
        return None
    except (RuntimeError, ValueError) as e:
        logger.error(f"PDF 페이지 수 조회 실패: {e}")
        return None
    except Exception as e:
        logger.error(f"PDF 페이지 수 조회 실패: {e}", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Navigation callbacks (module-level to avoid per-rerun recreation)
# ---------------------------------------------------------------------------


def _on_prev_click():
    """이전 페이지로 이동 (네비게이션 버튼 콜백)"""
    current = SessionManager.get("current_page", 1)
    new_page = max(1, current - 1)
    SessionManager.set("current_page", new_page)
    SessionManager.set("manual_nav_ts", time.time())
    st.session_state["pdf_nav_input_v6"] = new_page


def _on_next_click(total_pages: int):
    """다음 페이지로 이동 (네비게이션 버튼 콜백)"""
    current = SessionManager.get("current_page", 1)
    new_page = min(total_pages, current + 1)
    SessionManager.set("current_page", new_page)
    SessionManager.set("manual_nav_ts", time.time())
    st.session_state["pdf_nav_input_v6"] = new_page


def _on_page_change():
    """페이지 번호 입력 변경 시 (number_input on_change 콜백)"""
    new_p = st.session_state.get("pdf_nav_input_v6")
    if new_p:
        SessionManager.set("current_page", new_p)
        SessionManager.set("manual_nav_ts", time.time())


def _on_next_click_callback():
    """다음 페이지 네비게이션 콜백 (module-level)"""
    pdf_path = SessionManager.get("pdf_file_path", "")
    if pdf_path:
        total = _get_pdf_total_pages(os.path.abspath(pdf_path))
        if not total:
            return
        _on_next_click(total)


# ---------------------------------------------------------------------------
# PDF state resolution
# ---------------------------------------------------------------------------


def _resolve_pdf_state() -> dict | None:
    """Resolve current PDF state from session.

    Returns dict with keys: pdf_path, file_hash, total_pages, current_page.
    Returns None if no PDF is loaded or PDF is invalid.
    Handles external page navigation (e.g., from chat references via pdf_target_page).

    pdf_target_page는 일회성 점프 토큰이다:
    - dict {"page": int, "source": "auto"|"manual", "ts": float} 형식.
    - source=="auto"인데 manual_nav_ts가 토큰의 ts보다 크면 사용자가 더 최근에
      수동 네비게이션한 것이므로 토큰을 폐기하고 점프하지 않는다.
    - 레거시 int 값도 수용한다 (source="manual" 취급).
    """
    pdf_path_raw = SessionManager.get("pdf_file_path")
    if not pdf_path_raw:
        return None

    pdf_path = os.path.abspath(pdf_path_raw)
    file_hash = SessionManager.get("file_hash", "none")

    # 캐시 계층(st.cache_data)이 예외를 재전파할 수 있으므로 호출부에서도
    # 이중 안전망을 둔다. get_pdf_total_pages 자체는 raise하지 않는다.
    try:
        total_pages = _get_pdf_total_pages(pdf_path)
    except Exception as e:
        logger.error(f"PDF 페이지 수 조회 중 오류: {e}", exc_info=True)
        total_pages = None

    if not total_pages:
        return None

    # Handle external page navigation (e.g., from chat references)
    target = SessionManager.get("pdf_target_page")
    if target is not None:
        if isinstance(target, dict):
            page = target.get("page")
            source = target.get("source", "manual")
            ts = float(target.get("ts", 0) or 0)
        else:  # 레거시 int 형식 호환
            page = int(target)
            source = "manual"
            ts = 0.0
        if page is None:
            # 형식 오류 토큰(page 누락): 폴링마다 재평가되지 않도록 폐기한다.
            SessionManager.delete("pdf_target_page")
            st.session_state.pop("pdf_target_page", None)
        elif (
            source == "auto" and float(SessionManager.get("manual_nav_ts", 0) or 0) > ts
        ):
            # 사용자가 자동 점프 토큰 설정 이후 더 최근에 수동 네비게이션함
            # → 토큰 폐기 (점프 없음), 정상 nav-input 분기로 폴스루.
            SessionManager.delete("pdf_target_page")
            st.session_state.pop("pdf_target_page", None)
        else:
            current_page = min(max(1, int(page)), total_pages)
            SessionManager.set("current_page", current_page)
            # pdf_target_page는 일회성 소비: 점프 적용 후 키를 삭제하여
            # 사용자가 수동 네비게이션으로 벗어나도 매 rerun마다 참조 페이지로
            # 되돌아가지 않도록 보장한다.
            SessionManager.delete("pdf_target_page")
            st.session_state.pop("pdf_target_page", None)
            st.session_state["pdf_nav_input_v6"] = current_page
            return {
                "pdf_path": pdf_path,
                "file_hash": file_hash,
                "total_pages": total_pages,
                "current_page": current_page,
            }

    if "pdf_nav_input_v6" in st.session_state:
        current_page = min(
            max(1, int(st.session_state["pdf_nav_input_v6"])), total_pages
        )
        SessionManager.set("current_page", current_page)
    else:
        current_page = min(max(1, SessionManager.get("current_page", 1)), total_pages)
        st.session_state["pdf_nav_input_v6"] = current_page

    return {
        "pdf_path": pdf_path,
        "file_hash": file_hash,
        "total_pages": total_pages,
        "current_page": current_page,
    }


# ---------------------------------------------------------------------------
# Fragment: PDF 뷰어 + 네비게이션 컨트롤 (단일 fragment)
# ---------------------------------------------------------------------------


@st.fragment(run_every=2.0)
def render_pdf_area():
    """PDF 뷰어 + 네비게이션 컨트롤을 렌더링하는 단일 fragment.

    run_every 폴링으로 백그라운드 스트리밍 완료(_finalize_pdf_side_effects)가
    기록한 pdf_target_page/pdf_annotations를 최대 2초 내에 소비하여 자동 점프와
    하이라이트를 화면에 반영한다. 뷰어와 컨트롤이 한 fragment이므로 컨트롤
    클릭도 뷰어를 함께 재실행한다.

    손상/지원 불가 PDF는 스크립트를 죽이지 않고 뷰어 영역 안에서 오류로
    격리하여 렌더링한다. 어떤 예외도 이 함수 밖으로 새어나가지 않는다.
    """
    try:
        state = _resolve_pdf_state()
        if state is None:
            pdf_path = SessionManager.get("pdf_file_path")
            if pdf_path and os.path.exists(os.path.abspath(str(pdf_path))):
                st.error(
                    "⚠️ PDF 파일을 열 수 없습니다. 파일이 손상되었거나 지원되지 않는 형식입니다."
                )
            else:
                st.info(MSG_PDF_VIEWER_NO_FILE)
            return
        _display_pdf_viewer(
            state["pdf_path"], state["current_page"], state["file_hash"]
        )
        _display_pdf_controls(state["current_page"], state["total_pages"])
    except Exception as e:
        logger.error(f"PDF 뷰어 영역 오류: {e}", exc_info=True)
        st.error(
            "⚠️ PDF 파일을 열 수 없습니다. 파일이 손상되었거나 지원되지 않는 형식입니다."
        )


def _display_pdf_viewer(pdf_path, current_page, file_hash):
    try:
        from streamlit_pdf_viewer import pdf_viewer  # lazy: PDF 표시 시에만 import

        pdf_bytes = _get_pdf_bytes(pdf_path)
        if not pdf_bytes:
            st.error("⚠️ PDF 데이터를 불러올 수 없습니다.")
            return

        raw_annotations = SessionManager.get("pdf_annotations", [])
        if isinstance(raw_annotations, dict):
            annotations = (
                raw_annotations.get("annotations", [])
                if raw_annotations.get("file_hash") == file_hash
                else []
            )
        else:
            annotations = raw_annotations  # legacy list 형식 호환
        viewer_key = pdf_viewer_key(file_hash, current_page)

        pdf_viewer(
            input=pdf_bytes,
            render_text=True,
            pages_to_render=[current_page],
            annotations=annotations,
            annotation_outline_size=2,
            scroll_behavior="instant",
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
