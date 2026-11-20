"""
PDF 뷰어 및 문서 관련 UI 컴포넌트.
@st.fragment를 사용하여 PDF 렌더링과 네비게이션이 전체 페이지 리런 없이 독립적으로 업데이트됩니다.
"""

import json
import logging
import os

import streamlit as st

from common.config import MSG_PDF_VIEWER_NO_FILE
from common.exceptions import PDFProcessingError
from common.pdf_utils import get_pdf_page_count, open_pdf_document
from common.utils import fast_hash, safe_cache_data
from core.session import SessionManager
from ui.components.common import (
    get_doc_metadata,
    navigate_to_page,
    show_pdf_error,
    ui_error,
)
from ui.strings import t
from ui.widget_keys import (
    MANUAL_NAV_TS_KEY,
    PDF_NAV_INPUT_KEY,
    PDF_TARGET_PAGE_KEY,
    pdf_viewer_key,
)

logger = logging.getLogger(__name__)


def _render_coord_cache_warnings() -> None:
    """좌표 캐시 읽기 실패 문서마다 per-source 경고를 출력합니다.

    coord_cache_error 마커는 document_hydrator가 실패한 파일 단위로 설정하며,
    세션 documents에서 감지해 사용자에게 '하이라이트를 불러올 수 없습니다'를
    명시적으로 보여준다. 다른 소스 렌더링을 막지 않는다.
    """
    documents = SessionManager.get("documents", []) or []
    if not documents:
        return
    failed = 0
    seen_sources: set[str] = set()
    for doc in documents:
        meta = get_doc_metadata(doc)
        if not meta.get("coord_cache_error"):
            continue
        source = meta.get("file_path") or meta.get("source") or "문서"
        if source in seen_sources:
            continue
        seen_sources.add(source)
        failed += 1
    if failed:
        st.warning(t("pdf_highlight_load_failed"))


def _get_pending_upload_bytes(
    session_id: str, expected_file_hash: str | None = None
) -> bytes | None:
    """디스크 플러시 전 세션 SSoT(``pending_upload_persist``)에서 업로드 바이트를 조회한다.

    Streamlit 런타임은 ``src/main.py`` 를 ``sys.modules["__main__"]`` / ``"main"`` 에
    등록하지 않으므로 모듈 탐색 대신 ``SessionManager`` 를 직접 읽는다.
    세션 저장소에 없을 때만 기존 테스트 목 호환을 위해 ``sys.modules`` 폴백을 수행한다.
    """
    pending = SessionManager.get("pending_upload_persist", None, session_id)
    if isinstance(pending, dict):
        pending_hash = pending.get("file_hash")
        if (
            expected_file_hash
            and pending_hash
            and str(pending_hash) != expected_file_hash
        ):
            return None
        data = pending.get("file_bytes")
        return data if isinstance(data, bytes) and data else None

    import sys

    for mod_name in ("__main__", "src.main", "main"):
        module = sys.modules.get(mod_name)
        getter = getattr(module, "get_pending_upload_bytes", None)
        if callable(getter):
            try:
                value = getter(session_id)
            except Exception:  # noqa: BLE001 - fallback accessor must be safe
                return None
            return value if isinstance(value, bytes) and value else None
    return None


def _extract_page_bytes(doc, page: int) -> bytes:
    """열려 있는 문서에서 단일 페이지를 고속 직렬화한다 (1-based page)."""
    import pymupdf as fitz

    if doc.page_count == 0 or page > doc.page_count:
        return b""
    new_doc = fitz.open()
    try:
        new_doc.insert_pdf(doc, from_page=page - 1, to_page=page - 1)
        # deflate=False로 불필요한 zlib 압축 CPU 소모 제거 (단일 페이지 고속 직렬화)
        return new_doc.tobytes(garbage=1, deflate=False)
    finally:
        new_doc.close()


@safe_cache_data(ttl=300, show_spinner=False)
def _read_pdf_bytes_cached(pdf_path: str, mtime_ns: int) -> bytes:
    """실제 파일이 존재하고 비어 있지 않을 때만 캐시되는 PDF 전체 바이트 리더.

    mtime_ns 를 키에 포함해 파일이 원자적으로 교체되면 새 엔트리로 읽는다.
    빈 결과는 예외로 승격해 ``st.cache_data`` 에 b"" 가 고정되지 않게 한다.
    """
    with open(pdf_path, "rb") as f:
        data = f.read()
    if not data:
        raise OSError(f"empty PDF file: {pdf_path}")
    return data


@safe_cache_data(ttl=300, show_spinner=False)
def _read_pdf_total_pages_cached(pdf_path: str, mtime_ns: int) -> int | None:
    """유효한 페이지 수만 캐시한다 (None/0 은 예외로 승격)."""
    pages = get_pdf_page_count(pdf_path)
    if not pages:
        raise ValueError(f"unreadable PDF: {pdf_path}")
    return pages


@safe_cache_data(ttl=300, show_spinner=False)
def _read_pdf_page_bytes_cached(
    pdf_path: str, file_hash: str, page: int, mtime_ns: int
) -> bytes:
    """단일 페이지 추출 결과 중 비어 있지 않은 값만 캐시한다."""
    _ = file_hash  # 캐시 키 일부(파일 교체/해시별 분리)로만 사용
    with open_pdf_document(pdf_path) as doc:
        data = _extract_page_bytes(doc, page)
    if not data:
        raise ValueError(f"page {page} extraction returned empty: {pdf_path}")
    return data


def _get_pdf_bytes(pdf_path: str) -> bytes:
    """PDF 전체 바이트를 반환한다 (파일 미존재/읽기 실패 시 b"").

    파일 존재와 mtime 을 먼저 확인한 뒤에만 캐시 함수를 호출한다. 따라서
    지연 저장 중(파일 없음)의 b"" 가 300초 캐시에 고정되지 않는다.
    """
    if not pdf_path:
        return b""
    try:
        stat = os.stat(pdf_path)
    except OSError:
        return b""
    if stat.st_size <= 0:
        return b""
    try:
        return _read_pdf_bytes_cached(pdf_path, stat.st_mtime_ns)
    except (OSError, PDFProcessingError) as e:
        logger.error(f"PDF 파일 읽기 실패 ({pdf_path}): {e}")
        return b""


def _get_pdf_total_pages(pdf_path: str) -> int | None:
    """PDF 총 페이지 수 반환. 열 수 없으면 None (절대 raise하지 않음).

    파일이 없으면 캐시 함수를 호출하지 않아 None 캐시 오염을 막는다.
    실제 열기/페이지 수 계산은 공용 ``get_pdf_page_count`` 에 위임한다.
    """
    if not pdf_path:
        return None
    try:
        stat = os.stat(pdf_path)
    except OSError:
        return None
    if stat.st_size <= 0:
        return None
    try:
        return _read_pdf_total_pages_cached(pdf_path, stat.st_mtime_ns)
    except (OSError, ValueError) as e:
        logger.error(f"PDF 페이지 수 조회 실패 ({pdf_path}): {e}")
        return None


def _get_pending_page_bytes(page: int, file_hash: str | None = None) -> bytes:
    """디스크 기록 전 메모리 업로드 바이트에서 단일 페이지를 추출한다."""
    session_id = SessionManager.get_session_id()
    pending = _get_pending_upload_bytes(session_id, expected_file_hash=file_hash)
    if not pending:
        return b""
    try:
        import pymupdf as fitz

        with fitz.open(stream=pending, filetype="pdf") as doc:
            return _extract_page_bytes(doc, page)
    except Exception as e:
        logger.warning(f"[PDF] 메모리 폴백 {page}페이지 추출 실패: {e}")
        return b""


def _get_pdf_page_bytes(pdf_path: str, file_hash: str, page: int) -> bytes:
    """1페이지 추출: 원본 doc.select 대신 insert_pdf 클로닝으로 타 페이지 객체 검사 오버헤드 제거.

    디스크 파일이 아직 없으면(지연 저장 중) 메모리 업로드 바이트로 폴백해
    저장 완료 전 페이지 이동도 정상 동작하게 한다.
    """
    if not pdf_path or page < 1:
        return b""
    try:
        stat = os.stat(pdf_path)
    except OSError:
        return _get_pending_page_bytes(page, file_hash=file_hash)
    if stat.st_size <= 0:
        return _get_pending_page_bytes(page, file_hash=file_hash)
    try:
        return _read_pdf_page_bytes_cached(pdf_path, file_hash, page, stat.st_mtime_ns)
    except Exception as e:
        logger.warning(f"[PDF] {page}페이지 추출 실패 ({pdf_path}): {e}")
        return b""


def _page_scoped_annotations(annotations: list, current_page: int) -> list[dict]:
    """현재 페이지 주석만 골라 1페이지 기준으로 변환한다.

    x, y, width, height, color 등 나머지 키는 손대지 않는다: 이 좌표들은
    fitz/PDF-point 기준 페이지 상대값이라 추출된 단일 페이지에서도 이미
    정확하므로, 바뀌는 것은 페이지 순서값뿐이다 (추출본에서는 1).
    입력 dict는 절대 변경하지 않고 복사본에 page=1을 담아 반환한다.
    """
    if not annotations:
        return []
    scoped: list[dict] = []
    for item in annotations:
        if not isinstance(item, dict):
            continue
        raw = item.get("page")
        if raw is None:
            continue
        try:
            page_no = int(raw)
        except (TypeError, ValueError):
            continue
        if page_no != current_page:
            continue
        moved = dict(item)
        moved["page"] = 1
        scoped.append(moved)
    return scoped


# ---------------------------------------------------------------------------
# Navigation callbacks (module-level to avoid per-rerun recreation)
# ---------------------------------------------------------------------------


def _navigate(delta: int, total_pages: int | None = None) -> None:
    """Move by ``delta`` pages (clamped), syncing state + nav input (D4/D5)."""
    current = int(SessionManager.get("current_page", 1))
    if total_pages is None:
        target = max(1, current + delta)
    else:
        target = min(total_pages, max(1, current + delta))
    navigate_to_page(target)


def _on_prev_click(total_pages: int | None = None) -> None:
    """이전 페이지로 이동 (네비게이션 버튼 콜백, cached total 사용)."""
    _navigate(-1, total_pages)


def _on_next_click_callback(total_pages: int | None = None) -> None:
    """다음 페이지 네비게이션 콜백 (module-level).

    total은 렌더 시 resolve된 캐시 값을 args로 전달받는다 — 콜백에서
    재조회하지 않아 렌더당 resolve 1회를 보장한다 (fragment deps 축소).
    """
    _navigate(1, total_pages)


def _on_page_change(
    widget_key_or_total: str | int | None = None,
    total_pages: int | None = None,
) -> None:
    """페이지 번호 입력 변경 시 (number_input on_change 콜백).

    - 위젯 연동 호출: _on_page_change(widget_key, total_pages)
    - 기존/테스트 호환 호출: _on_page_change(total_pages)
    """
    raw = None
    total = None

    if isinstance(widget_key_or_total, str):
        # 신규 규격: 위젯 키를 직접 전달받아 해당 위젯의 변경값을 즉시 획득
        raw = st.session_state.get(widget_key_or_total)
        total = total_pages
    elif isinstance(widget_key_or_total, int):
        # 레거시 및 단위 테스트 호환: 단일 정수 인자가 total_pages로 들어온 경우
        total = widget_key_or_total
        current = SessionManager.get("current_page", 1)
        expected_key = f"{PDF_NAV_INPUT_KEY}_{current}"
        raw = st.session_state.get(expected_key)
        if raw is None:
            raw = st.session_state.get(PDF_NAV_INPUT_KEY)
    else:
        total = total_pages
        current = SessionManager.get("current_page", 1)
        expected_key = f"{PDF_NAV_INPUT_KEY}_{current}"
        raw = st.session_state.get(expected_key)
        if raw is None:
            raw = st.session_state.get(PDF_NAV_INPUT_KEY)

    if total is None or raw is None:
        return
    try:
        new_page = int(raw)
    except (TypeError, ValueError):
        return
    navigate_to_page(min(total, max(1, new_page)))


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
      (uiux-fix-p1 INT-1: "auto" 소스는 더 이상 생성되지 않음 — 자동 점프 제거.
      수동 참조 버튼(chat.py)이 "manual" 토큰만 생성한다.)
    - source=="auto"인데 manual_nav_ts가 토큰의 ts보다 크면 사용자가 더 최근에
      수동 네비게이션한 것이므로 토큰을 폐기하고 점프하지 않는다.
      (레거시 "auto" 토큰과 방어적 처리를 위해 로직은 유지)
    - 레거시 int 값도 수용한다 (source="manual" 취급).
    """
    sid = SessionManager.get_session_id()
    pdf_path_raw = SessionManager.get("pdf_file_path", session_id=sid)
    if not pdf_path_raw:
        # PDF를 소비할 수 없는 상태: 일회성 점프 토큰이 남아 이후 PDF 로드 시
        # 옛 문서 페이지로 자동 점프하지 않도록 self-clean 한다.
        SessionManager.delete(PDF_TARGET_PAGE_KEY, session_id=sid)
        st.session_state.pop(PDF_TARGET_PAGE_KEY, None)
        return None

    pdf_path = os.path.abspath(pdf_path_raw)
    file_hash = SessionManager.get("file_hash", "none", session_id=sid)

    # [최적화] 1. 세션에 캐시된 total_pages 확인
    cached_total = SessionManager.get("total_pages", session_id=sid)
    total_pages = (
        cached_total if isinstance(cached_total, int) and cached_total > 0 else None
    )

    # [FAST-PATH 우선순위 보장] 1페이지 미리보기가 유효하면 디스크 I/O를 기다리지 않고 우선 1페이지 허용
    preview = SessionManager.get("first_page_preview", session_id=sid)
    has_valid_preview = (
        isinstance(preview, dict)
        and preview.get("file_hash") == file_hash
        and bool(preview.get("bytes"))
    )

    if not total_pages:
        try:
            # _get_pdf_total_pages는 @safe_cache_data로 캐싱되어 0.5ms 미만 소요
            total_pages = _get_pdf_total_pages(pdf_path)
            if total_pages:
                SessionManager.set("total_pages", total_pages, session_id=sid)
        except Exception as e:
            logger.error(f"PDF 페이지 수 조회 중 오류: {e}", exc_info=True)
            total_pages = None

        # 실제 파일로부터 페이지 수 조회가 불가능한 극단적 경우에만 1페이지 미리보기로 폴백
        if not total_pages and has_valid_preview:
            total_pages = 1
            logger.info(
                "[PDF] total_pages 산출 지연/실패로 1페이지 미리보기를 통해 임시 확정"
            )

    if not total_pages:
        SessionManager.delete(PDF_TARGET_PAGE_KEY, session_id=sid)
        st.session_state.pop(PDF_TARGET_PAGE_KEY, None)
        return None

    # Handle external page navigation (e.g., from chat references)
    target = SessionManager.get(PDF_TARGET_PAGE_KEY, session_id=sid)
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
            SessionManager.delete(PDF_TARGET_PAGE_KEY, session_id=sid)
            st.session_state.pop(PDF_TARGET_PAGE_KEY, None)
        elif (
            source == "auto"
            and float(SessionManager.get(MANUAL_NAV_TS_KEY, 0, session_id=sid) or 0)
            > ts
        ):
            # 사용자가 자동 점프 토큰 설정 이후 더 최근에 수동 네비게이션함
            # → 토큰 폐기 (점프 없음), 정상 nav-input 분기로 폴스루.
            SessionManager.delete(PDF_TARGET_PAGE_KEY, session_id=sid)
            st.session_state.pop(PDF_TARGET_PAGE_KEY, None)
        else:
            current_page = min(max(1, int(page)), total_pages)
            SessionManager.set("current_page", current_page, session_id=sid)
            # pdf_target_page는 일회성 소비: 점프 적용 후 키를 삭제하여
            # 사용자가 수동 네비게이션으로 벗어나도 매 rerun마다 참조 페이지로
            # 되돌아가지 않도록 보장한다. 점프한 페이지를 네비 입력 위젯에도
            # 동기화한다: navigate_to_page()가 콜백 단계에서 위젯 키를 쓰지만,
            # 토큰 소비 경로는 콜백을 거치지 않으므로 여기서 직접 맞춘다.
            # 위젯 생성 이전의 대입이므로 Streamlit 규약상 허용된다.
            SessionManager.delete(PDF_TARGET_PAGE_KEY, session_id=sid)
            st.session_state.pop(PDF_TARGET_PAGE_KEY, None)
            # st.session_state[PDF_NAV_INPUT_KEY] 직접 대입 제거 (Streamlit 1.60 규격 준수:
            # 위젯 생성 전/후 직접 세션 위젯 키 수정 대신 세션 관리자의 current_page와
            # render_pdf_controls 내부의 상태 분기값에 의존하여 stale/StreamlitAPIException 방지)
            return {
                "pdf_path": pdf_path,
                "file_hash": file_hash,
                "total_pages": total_pages,
                "current_page": current_page,
            }

    # 위젯 상태(sticky key)보다 매니저(SessionManager.current_page)가 우선한다:
    # 모든 네비게이션(버튼/입력/참조점프/새파일)은 단일 navigate_to_page() 경로로
    # 매니저에 먼저 기록되므로, 오래된 위젯 값이 매니저를 덮어쓰면 버튼 이동이
    # 무효화된다. 사용자 입력은 on_change 콜백이 매니저에 동기화하므로 여기서
    # 다시 읽을 필요가 없으며, 위젯 default는 렌더 시 매니저에서 받는다.
    try:
        managed_page = int(SessionManager.get("current_page", 1, session_id=sid) or 1)
    except (TypeError, ValueError):
        managed_page = 1
    current_page = min(max(1, managed_page), total_pages)
    if managed_page != current_page:
        SessionManager.set("current_page", current_page, session_id=sid)

    return {
        "pdf_path": pdf_path,
        "file_hash": file_hash,
        "total_pages": total_pages,
        "current_page": current_page,
    }


# ---------------------------------------------------------------------------
# Fragment: PDF 뷰어 + 네비게이션 컨트롤 (단일 fragment)
# ---------------------------------------------------------------------------


@st.fragment()
def render_pdf_area():
    """PDF 뷰어 + 네비게이션 컨트롤을 렌더링하는 단일 fragment.

    run_every 폴링 없이 동작한다(원래 2s 폴링의 소비 대상 재분석 결과):
    - pdf_annotations: 답변 완료 시 `_finalize_pdf_side_effects` 또는 비동기
      prewarm 콜백(`_make_prewarm_done_callback`)이 현재 페이지 주석 변경을
      감지해 필요 시에만 리런을 트리거한다.
    - pdf_target_page(수동 참조 점프): on_click 콜백(_handle_page_jump/_handle_doc_jump)이
      설정하고, 버튼 클릭이 유발하는 다음 전체 rerun에서 _resolve_pdf_state가 소비한다.
      Callback-phase st.rerun()은 no-op 경고를 유발하므로 사용하지 않는다.
    - 페이지/current_page: 네비 컨트롤의 on_click/on_change 콜백이 fragment rerun을
      발생시키고, `pdf_viewer_key`의 페이지 키 변경으로 컴포넌트가 remount된다.
    따라서 모든 뷰어 갱신은 명시적 rerun 경로로 전달되며, 유휴 폴링 타이머가 없어
    무조건 재렌더(2s마다 PDF 재전송)를 제거한다. (fix-001-polling-fragments)
    뷰어와 컨트롤이 한 fragment이므로 컨트롤 클릭도 뷰어를 함께 재실행한다.

    손상/지원 불가 PDF는 스크립트를 죽이지 않고 뷰어 영역 안에서 오류로
    격리하여 렌더링한다. 어떤 예외도 이 함수 밖으로 새어나가지 않는다.
    """
    try:
        state = _resolve_pdf_state()
        _render_coord_cache_warnings()
        if state is None:
            pdf_path = SessionManager.get("pdf_file_path")
            if pdf_path and os.path.exists(os.path.abspath(str(pdf_path))):
                show_pdf_error("open")
            else:
                st.info(MSG_PDF_VIEWER_NO_FILE)
            return
        render_pdf_viewer(state["pdf_path"], state["current_page"], state["file_hash"])
        render_pdf_controls(state["current_page"], state["total_pages"])
    except Exception as e:
        logger.error(f"PDF 뷰어 영역 오류: {e}", exc_info=True)
        show_pdf_error("open")


def render_pdf_viewer(pdf_path, current_page, file_hash):
    try:
        from streamlit_pdf_viewer import pdf_viewer  # lazy: PDF 표시 시에만 import

        sid = SessionManager.get_session_id()
        raw_annotations = SessionManager.get("pdf_annotations", [], session_id=sid)
        if isinstance(raw_annotations, dict):
            annotations = (
                raw_annotations.get("annotations", [])
                if raw_annotations.get("file_hash") == file_hash
                else []
            )
        else:
            annotations = raw_annotations  # legacy list 형식 호환
        if not isinstance(annotations, list):
            annotations = []
        scoped_annotations = _page_scoped_annotations(annotations, current_page)
        annotations_sig = (
            fast_hash(
                json.dumps(scoped_annotations, sort_keys=True, default=str),
                length=8,
            )
            if scoped_annotations
            else ""
        )
        viewer_key = pdf_viewer_key(file_hash, current_page, annotations_sig)

        # [FAST-PATH] 1페이지는 업로드 검증 단계에서 선행 추출된 메모리
        # 바이트가 있으면 디스크 재오픈/재파싱 없이 즉시 렌더링한다.
        page_bytes = b""
        if current_page == 1:
            preview = SessionManager.get("first_page_preview", session_id=sid)
            if (
                isinstance(preview, dict)
                and preview.get("file_hash") == file_hash
                and isinstance(preview.get("bytes"), bytes)
                and preview["bytes"]
            ):
                page_bytes = preview["bytes"]

        # 단일 페이지 우선: 전체 문서 재전송을 피한다.
        if not page_bytes:
            page_bytes = _get_pdf_page_bytes(pdf_path, file_hash, current_page)
        if page_bytes:
            pdf_viewer(
                input=page_bytes,
                render_text=True,
                pages_to_render=[1],
                annotations=_page_scoped_annotations(annotations, current_page),
                annotation_outline_size=2,
                scroll_behavior="instant",
                key=viewer_key,
            )
            return

        pdf_bytes = _get_pdf_bytes(pdf_path)
        if not pdf_bytes:
            show_pdf_error("data")
            return

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
        ui_error(t("pdf_error_viewer"))
    except Exception as e:
        logger.error(f"PDF 뷰어 렌더링 오류: {e}", exc_info=True)
        ui_error(t("pdf_error_render"))


def render_pdf_controls(current_page, total_pages):
    try:
        # 1. current_page의 유효 범위를 1 ~ total_pages로 엄격히 제한
        current_page = min(max(1, int(current_page)), total_pages)

        col_prev, col_page, col_next = st.columns(
            [1, 2, 1], gap="small", vertical_alignment="center"
        )

        with col_prev:
            st.button(
                t("nav_prev"),
                width="stretch",
                key="btn_nav_prev_v6",
                disabled=current_page <= 1,
                on_click=_on_prev_click,
                args=(total_pages,),
                help=t("pdf_page_help"),
            )
        with col_page:
            dyn_key = f"{PDF_NAV_INPUT_KEY}_{current_page}"
            # 이전 페이지 번호 입력 위젯 키 정리 (메모리 누수 및 세션 상태 잔재 방지)
            for k in list(st.session_state.keys()):
                if str(k).startswith(f"{PDF_NAV_INPUT_KEY}_") and k != dyn_key:
                    st.session_state.pop(k, None)

            st.number_input(
                t("pdf_page_label"),
                min_value=1,
                max_value=total_pages,
                value=current_page,
                key=dyn_key,
                on_change=_on_page_change,
                args=(total_pages,),
                label_visibility="visible",
                help=t("pdf_page_help"),
            )

        with col_next:
            st.button(
                t("nav_next"),
                width="stretch",
                key="btn_nav_next_v6",
                disabled=current_page >= total_pages,
                help=t("pdf_page_help"),
                on_click=_on_next_click_callback,
                args=(total_pages,),
            )

    except Exception as e:
        logger.error(f"PDF 컨트롤바 오류: {e}", exc_info=True)
