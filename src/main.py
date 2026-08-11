# src/main.py
# Streamlit 기반 RAG 챗봇의 메인 진입점 및 전체 오케스트레이션을 담당하는 파일
"""
RAG Chatbot 애플리케이션의 메인 진입점 파일입니다.
Streamlit 프레임워크를 기반으로 UI를 구성하고 세션 상태를 관리합니다.
"""

# P4 (조사: .omo/evidence/log-issue-fixes/p4_warning_source.md): langgraph 1.0.1의
# LangChainPendingDeprecationWarning("allowed_objects 기본값 변경 예정")은 모듈 임포트
# 시점(langgraph/checkpoint/serde/jsonplus.py 모듈 최상위 LC_REVIVER = Reviver())에
# 발생하며 앱의 JsonPlusSerializer 파라미터로는 제거할 수 없다. 경고 필터는
# list-index-0 우선(last-registered wins)이므로, langchain_core._api.deprecation이
# 임포트 시 자체 등록하는 필터보다 우리의 ignore가 나중에 등록(먼저 검사)되어야 한다.
# 따라서 (1) 해당 모듈을 먼저 임포트한 뒤 (2) 카테고리+메시지 범위로만 억제를
# 등록한다 — 다른 경고는 모두 그대로 유지된다.
import warnings

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

warnings.filterwarnings(
    "ignore",
    message="The default value of .allowed_objects. will change",
    category=LangChainPendingDeprecationWarning,
)

# [Lazy Import용] 런타임에 필요한 모듈들
import asyncio
import atexit
import contextlib
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Literal, cast

import streamlit as st

from common.async_worker import AsyncWorker
from common.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_MODEL,
    MSG_ERROR_OLLAMA_NOT_RUNNING,
)
from common.constants import FilePathConstants, StringConstants
from common.logging_config import setup_logging
from common.utils import safe_cache_resource

# 1. Streamlit 페이지 설정 (최우선 실행 - 가이드라인 준수)
st.set_page_config(
    page_title=StringConstants.PAGE_TITLE,
    layout=cast(Literal["centered", "wide"], StringConstants.LAYOUT),
    initial_sidebar_state="expanded",
)


# 2. 로깅 설정 (최상단, 서버당 1회만 초기화)
@st.cache_resource
def _init_logging() -> logging.Logger:
    return setup_logging(log_level="INFO", log_file=FilePathConstants.LOG_FILE)


logger = _init_logging()

MAX_FILE_SIZE_MB = StringConstants.MAX_FILE_SIZE_MB

# NumExpr 최대 스레드 설정 (CPU 코어 수 기반)
_numexpr_threads = os.cpu_count() or 4
os.environ.setdefault("NUMEXPR_MAX_THREADS", str(_numexpr_threads))

# 비동기 워커 초기화 (서버 인스턴스당 단일 이벤트 루프, nest_asyncio 대체)
_async_worker = AsyncWorker()

# current_page / is_generating_answer are initialized via DEFAULT_SESSION_STATE
# and synced by UIBridge.sync_session(). pdf_target_page는 일회성 점프 토큰(dict
# 또는 None)으로 DEFAULT_SESSION_STATE에 포함되지 않으므로 여기서 초기화한다.
if "pdf_target_page" not in st.session_state:
    st.session_state.pdf_target_page = None


def _check_windows_integrity():
    """
    [Background] Windows 환경의 라이브러리 충돌을 체크하고 주기적으로 세션을 정리합니다.
    """
    try:
        from core.session import SessionManager

        # 1시간 이상 활동 없는 세션 정리 (물리적 파일 삭제 및 참조 해제 포함)
        SessionManager.cleanup_expired_sessions(max_idle_seconds=3600)
    except Exception as e:
        logger.error(f"[SYSTEM] [CLEANUP] 세션 정리 중 오류: {e}")

    import platform

    if platform.system() != "Windows" or os.getenv("GITHUB_ACTIONS") == "true":
        return

    try:
        import importlib.util

        if importlib.util.find_spec("torch") is None:
            logger.warning("[SYSTEM] [INTEGRITY] torch 모듈을 찾을 수 없습니다.")
            return

        with contextlib.suppress(ValueError, RuntimeError):
            logger.info("[SYSTEM] [INTEGRITY] Windows 라이브러리 무결성 점검 완료 (OK)")

    except Exception as e:
        logger.warning(f"[SYSTEM] [INTEGRITY] 점검 중 예외 발생: {e}")


@st.cache_resource
def _start_global_background_worker():
    """
    [Singleton] 서버 인스턴스당 단 하나만 실행되는 백그라운드 워커입니다.
    세션 정리 및 시스템 무결성 점검을 주기적으로 수행합니다.
    """

    def maintenance_loop():
        logger.info("[SYSTEM] 전역 백그라운드 워커 시작됨")
        while True:
            with contextlib.suppress(Exception):
                _check_windows_integrity()
            # 1시간(3600초) 대기 후 반복
            time.sleep(3600)

    thread = threading.Thread(
        target=maintenance_loop, name="GlobalMaintenanceWorker", daemon=True
    )
    thread.start()
    return thread


@st.cache_data(ttl=300)
def _get_available_models_cached():
    """Ollama 모델 목록을 캐싱하여 UI 블로킹을 최소화합니다."""
    from core.model_loader import get_available_models

    return get_available_models()


def _load_available_models() -> list[str]:
    """Ollama 모델 목록을 동적으로 조회하여 반환합니다.

    get_available_models는 Ollama 미연결 시 기본 모델과 오류 문자열을 함께
    반환할 수 있으므로, UI에 노출되면 안 되는 오류 문자열은 걸러냅니다.
    조회가 실패(타임아웃/비정상 종료 등)하면 안전하게 기본 모델로 폴백합니다.
    """
    try:
        models = _get_available_models_cached()
    except Exception:
        logger.exception("[SYSTEM] Ollama 모델 목록 조회 실패, 기본 모델로 대체")
        return [DEFAULT_OLLAMA_MODEL]

    filtered = [m for m in models if m != MSG_ERROR_OLLAMA_NOT_RUNNING]
    return filtered or [DEFAULT_OLLAMA_MODEL]


@safe_cache_resource(show_spinner=False)
def _init_temp_directory():
    """임시 디렉토리를 초기화합니다."""
    temp_path = Path(FilePathConstants.TEMP_DIR).absolute()
    temp_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"[SYSTEM] [INIT] 임시 디렉토리 준비 완료: {temp_path}")
    return str(temp_path)


def _cleanup_current_file():
    """모든 세션에서 사용 중인 임시 파일을 삭제합니다. (종료 핸들러용)

    "default" 세션만 정리하면 다른 세션의 임시 PDF가 남으므로,
    전체 세션의 경로를 수집하여 삭제합니다.
    """
    from core.session import SessionManager

    try:
        paths = SessionManager.get_all_pdf_paths()
        for path in paths:
            SessionManager.safe_remove_file(path)
    except Exception:
        pass


@st.cache_resource
def _register_cleanup_handlers():
    """[Singleton] 프로세스 종료 핸들러를 단 한 번만 등록합니다."""
    atexit.register(_cleanup_current_file)
    logger.info("[SYSTEM] 프로세스 종료 핸들러 등록 완료")
    return True


# 앱 시작 시 초기화 수행 (캐싱으로 인해 최초 1회만 작동)
_init_temp_directory()
_start_global_background_worker()
_register_cleanup_handlers()


async def _bg_rebuild_task(
    session_id: str, file_path: str, file_name: str, embedder_name: str
):
    """
    [Background Task] 업로드된 파일과 선택된 모델을 사용하여 RAG 파이프라인을 비동기로 재구축합니다.
    """
    from core.model_loader import ModelManager
    from core.rag_core import RAGSystem
    from core.session import SessionManager

    SessionManager.set_session_id(session_id)
    SessionManager.set("rebuild_done", False, session_id=session_id)
    SessionManager.set("rebuild_error", None, session_id=session_id)
    SessionManager.set(
        "rebuild_status", f"'{file_name}' 분석 중...", session_id=session_id
    )

    # 타임라인에 분석 시작 메시지 추가 (하나의 메시지를 업데이트)
    # on_file_upload가 만든 자리표시자(msg_id=build_{session_id})를 재사용하여
    # 빈 대화 깜빡임 없이 진행 상황이 그 위에 업데이트된다.
    build_msg_id = f"build_{session_id}"
    SessionManager.set("build_msg_id", build_msg_id, session_id=session_id)
    SessionManager.add_message(
        "system",
        f"📄 '{file_name}' 문서 분석 시작",
        msg_type="build_progress",
        msg_id=build_msg_id,
        progress=0,
        status="분석 준비 중...",
        cancelable=True,
        logs=[],
        session_id=session_id,
    )

    try:
        embedder = await ModelManager.get_embedder(embedder_name)

        # Check cancellation before starting build
        if SessionManager.get("rebuild_cancelled", False, session_id=session_id):
            logger.info(f"[MAIN] Rebuild cancelled by user for session {session_id}")
            SessionManager.set("rebuild_cancelled", False, session_id=session_id)
            SessionManager.set("rebuild_progress", 0, session_id=session_id)
            SessionManager.add_status_log(
                "❌ 문서 분석이 취소되었습니다.", session_id=session_id
            )
            SessionManager.add_message(
                "system",
                "문서 분석이 취소되었습니다",
                msg_type="build_error",
                msg_id=build_msg_id,
                error="사용자가 분석을 취소함",
                session_id=session_id,
            )
            return

        rag_sys = RAGSystem(session_id=session_id)

        def _report_progress(pct: int, msg: str = ""):
            SessionManager.set("rebuild_progress", pct, session_id=session_id)
            if msg:
                SessionManager.set("rebuild_status", msg, session_id=session_id)
            # 타임라인 진행 메시지 업데이트 (동일 msg_id)
            SessionManager.add_message(
                "system",
                f"📄 '{file_name}' 분석 진행 중",
                msg_type="build_progress",
                msg_id=build_msg_id,
                progress=pct,
                status=msg or f"진행률 {pct}%",
                cancelable=True,
                logs=SessionManager.get("status_logs", [], session_id) or [],
                session_id=session_id,
            )

        SessionManager.set("rebuild_progress", 0, session_id=session_id)

        # Check cancellation again before expensive build_pipeline call
        if SessionManager.get("rebuild_cancelled", False, session_id=session_id):
            logger.info(f"[MAIN] Rebuild cancelled by user for session {session_id}")
            SessionManager.set("rebuild_cancelled", False, session_id=session_id)
            SessionManager.set("rebuild_progress", 0, session_id=session_id)
            SessionManager.add_status_log(
                "❌ 문서 분석이 취소되었습니다.", session_id=session_id
            )
            SessionManager.add_message(
                "system",
                "문서 분석이 취소되었습니다",
                msg_type="build_error",
                msg_id=build_msg_id,
                error="사용자가 분석을 취소함",
                session_id=session_id,
            )
            return

        def _is_cancelled() -> bool:
            return bool(
                SessionManager.get("rebuild_cancelled", False, session_id=session_id)
            )

        success_message, cache_used = await rag_sys.build_pipeline(
            file_path=file_path,
            file_name=file_name,
            embedder=embedder,
            on_progress=_report_progress,
            check_cancelled=_is_cancelled,
        )

        SessionManager.set("rebuild_progress", 100, session_id=session_id)
        SessionManager.set("pdf_processed", True, session_id=session_id)
        SessionManager.set("pdf_processing_error", None, session_id=session_id)
        SessionManager.set(
            "doc_stats",
            {
                "file_name": file_name,
                "cache_used": bool(cache_used),
                "embedder": embedder_name,
            },
            session_id=session_id,
        )
        SessionManager.add_status_log(f"✅ {success_message}", session_id=session_id)
        # 타임라인 진행 메시지 완료 처리 (동일 msg_id)
        SessionManager.add_message(
            "system",
            f"✅ {success_message}",
            msg_type="build_progress",
            msg_id=build_msg_id,
            progress=100,
            status=success_message,
            cancelable=False,
            done=True,
            logs=[],
            session_id=session_id,
        )
        SessionManager.add_message("system", success_message, session_id=session_id)
    except asyncio.CancelledError:
        logger.info(
            f"[MAIN] Rebuild pipeline cancelled mid-build for session {session_id}"
        )
        SessionManager.set("rebuild_cancelled", False, session_id=session_id)
        SessionManager.set("rebuild_progress", 0, session_id=session_id)
        SessionManager.add_message(
            "system",
            "문서 분석이 취소되었습니다",
            msg_type="build_error",
            msg_id=build_msg_id,
            error="비동기 작업 취소됨",
            session_id=session_id,
        )
    except Exception as e:
        logger.error(f"Background RAG rebuild error: {e}", exc_info=True)
        error_msg = f"문서 처리 중 오류가 발생했습니다: {str(e)}"
        SessionManager.set("rebuild_error", error_msg, session_id=session_id)
        SessionManager.set("pdf_processing_error", error_msg, session_id=session_id)
        SessionManager.set("rebuild_progress", 0, session_id=session_id)
        SessionManager.set("pdf_processed", False, session_id=session_id)
        SessionManager.add_message(
            "system",
            "문서 분석 중 오류 발생",
            msg_type="build_error",
            msg_id=build_msg_id,
            error=error_msg,
            session_id=session_id,
        )
        SessionManager.add_message("assistant", error_msg, session_id=session_id)
    finally:
        SessionManager.set("rebuild_done", True, session_id=session_id)
        SessionManager.set("is_building_rag", False, session_id=session_id)


def _update_qa_chain(session_id: str | None = None) -> None:
    """
    문서 인덱싱은 유지한 채 LLM(QA Chain)만 교체합니다.
    """
    from core.session import SessionManager

    sid = session_id or SessionManager.get_session_id()
    selected_model = SessionManager.get("last_selected_model", session_id=sid)
    try:
        SessionManager.add_status_log("🔄 추론 모델 교체 중", session_id=sid)
        from core.model_loader import load_llm

        model_name = str(selected_model or DEFAULT_OLLAMA_MODEL)
        llm = load_llm(model_name)
        SessionManager.set("llm", llm, session_id=sid)
        SessionManager.add_status_log("✅ 추론 모델 교체 완료", session_id=sid)
    except Exception as e:
        error_msg = f"QA 체인 업데이트 실패: {e}"
        logger.error(f"QA 업데이트 실패: {e}", exc_info=True)
        SessionManager.add_status_log(f"❌ {error_msg}", session_id=sid)
        SessionManager.add_message("assistant", error_msg, session_id=sid)
    finally:
        SessionManager.set("rag_build_complete_flag", True, session_id=sid)


def _post_upload_error(error_msg: str, session_id: str | None = None) -> None:
    """업로드 검증 실패를 타임라인 메시지로 전달합니다 (메인 레벨 요소 삽입 금지).

    on_file_upload는 _render_app_layout보다 먼저 실행되므로 st.error(...)는
    메인 블록 상단(두 컬럼 위)에 렌더되어 채팅 스크롤러를 아래로 밀어낸다.
    실패 메시지는 세션 상태에 기록하고 chat 컬럼 내부의 타임라인 fragment가
    렌더링하도록 해 레이아웃을 그대로 유지한다. (QA-실패 패턴과 동일)
    """
    from core.session import SessionManager

    sid = session_id or SessionManager.get_session_id()
    # error_msg는 이미 "❌ " 접두사를 포함하므로 중복해서 붙이지 않는다.
    SessionManager.add_status_log(error_msg, session_id=sid)
    SessionManager.add_message("assistant", error_msg, session_id=sid)


def on_file_upload() -> None:
    from core.document_processor import compute_file_hash
    from core.session import SessionManager
    from infra.notification_system import SystemNotifier

    uploaded_file = st.session_state.get("pdf_uploader")
    if uploaded_file is None or not hasattr(uploaded_file, "type"):
        return

    if uploaded_file.type != "application/pdf":
        _post_upload_error(
            "❌ 올바른 PDF 파일이 아닙니다. PDF 형식의 파일을 업로드해주세요."
        )
        return

    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        _post_upload_error(
            f"❌ 파일 크기가 너무 큽니다 ({file_size_mb:.2f} MB). {MAX_FILE_SIZE_MB}MB 이하의 파일을 업로드해주세요."
        )
        return

    last_file_name = SessionManager.get("last_uploaded_file_name")
    last_file_hash = SessionManager.get("file_hash")

    file_bytes = b""
    try:
        # uploaded_file는 BytesIO와 유사한 객체이므로 포인터를 리셋하여 전체 바이트를 계산합니다.
        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)
        uploaded_hash = compute_file_hash("", data=file_bytes)
    except Exception:
        uploaded_hash = ""

    # [확장자 위조 방지] .pdf 확장자지만 실제로는 PDF가 아닌 바이트를 차단한다.
    # 손상된 파일이 temp에 복사되거나 빌드가 시작되면 뷰어/파이프라인이 크래시
    # 하므로, 여기서 즉시 검증하고 실패 시 아무 상태도 바꾸지 않는다.
    try:
        import fitz  # lazy import: 검증이 필요한 시점에만 로드

        if not file_bytes:
            raise ValueError("PDF 파일이 비어 있습니다.")
        with fitz.open(stream=file_bytes, filetype="pdf") as upload_doc:
            page_count = len(upload_doc)
        if page_count < 1:
            raise ValueError("PDF 파일에 페이지가 없습니다.")
    except Exception as e:
        logger.warning(f"손상되었거나 읽을 수 없는 PDF 업로드 차단: {e}")
        _post_upload_error(
            "❌ 손상되었거나 읽을 수 없는 PDF 파일입니다. 올바른 PDF 파일을 업로드해주세요."
        )
        return

    if uploaded_file.name != last_file_name or uploaded_hash != last_file_hash:
        sid = SessionManager.get_session_id()
        # [UX] 새 파일 전환: 대화/플래그를 렌더 전(콜백 시점)에 초기화하여
        # 업로드 직후 불필요한 전체 rerun 없이도 빈 화면 깜빡임이 사라진다.
        # (reset_for_new_file이 current_page/pdf_annotations/pdf_target_page 초기화 포함)
        SessionManager.reset_for_new_file(session_id=sid)
        # 새 문서는 항상 1페이지부터 열이도록 PDF 네비게이션 위젯 상태와
        # 일회성 점프 키(pdf_target_page)를 초기화합니다. pdf_nav_input_v6는
        # INTERACTIVE_KEYS에 속해 스냅샷/복원되므로 직접 session_state를
        # 갱신하여 이전 문서의 마지막 페이지 값이 재사용되지 않게 합니다.
        st.session_state["pdf_nav_input_v6"] = 1
        st.session_state.pop("pdf_target_page", None)
        old_path = SessionManager.get("pdf_file_path")
        if old_path:
            SessionManager.safe_remove_file(old_path)

        SessionManager.set("last_uploaded_file_name", uploaded_file.name)
        SessionManager.set("file_hash", uploaded_hash)

        try:
            temp_dir = os.path.abspath(FilePathConstants.TEMP_DIR)
            sid = SessionManager.get_session_id()
            tmp_path = os.path.join(temp_dir, f"upload_{sid}_{int(time.time())}.pdf")

            with open(tmp_path, "wb") as f:
                shutil.copyfileobj(uploaded_file, f)

            SessionManager.set("pdf_file_path", tmp_path)
            # [UX] 분석 진행 자리표시자 메시지: _bg_rebuild_task가 동일 msg_id로 업데이트
            SessionManager.add_message(
                "system",
                f"📄 '{uploaded_file.name}' 문서 분석 시작",
                msg_type="build_progress",
                msg_id=f"build_{sid}",
                progress=0,
                status="분석 준비 중...",
                cancelable=True,
                logs=[],
                session_id=sid,
            )
            SessionManager.set("new_file_uploaded", True)
            SystemNotifier.success(f"문서 업로드 완료: {uploaded_file.name}")
        except Exception as e:
            SystemNotifier.error("파일 저장 중 오류 발생", details=str(e))
    else:
        SystemNotifier.success(
            f"'{uploaded_file.name}'은(는) 이미 업로드된 동일한 문서입니다."
        )


def on_new_chat() -> None:
    """새 대화 시작: 문서/파이프라인은 유지하고 대화만 초기화합니다."""
    from core.session import SessionManager

    current_sid = SessionManager.get_session_id()
    SessionManager.reset_conversation(current_sid)
    st.rerun()


def on_model_change() -> None:
    from core.session import SessionManager

    selected = st.session_state.get("model_selector")
    last = SessionManager.get("last_selected_model")

    if not selected or "---" in selected or selected == last:
        return

    SessionManager.set("last_selected_model", selected)
    if SessionManager.get("pdf_processed"):
        SessionManager.set("needs_qa_chain_update", True)


def on_embedding_change() -> None:
    from core.session import SessionManager

    selected = st.session_state.get("embedding_model_selector")
    last = SessionManager.get("last_selected_embedding_model")

    if not selected or selected == last:
        return

    SessionManager.set("last_selected_embedding_model", selected)
    if SessionManager.get("pdf_file_path"):
        SessionManager.set("needs_rag_rebuild", True)


def on_refresh_models() -> None:
    """Ollama 모델 목록을 재조회하여 사이드바 셀렉터를 갱신합니다."""
    _get_available_models_cached.clear()
    st.session_state.available_models_list = _load_available_models()
    st.rerun()


def _render_app_layout(available_models: list[str] | None = None) -> None:
    from core.session import SessionManager
    from ui.components.sidebar import render_settings_content

    with st.sidebar:
        render_settings_content(
            file_uploader_callback=on_file_upload,
            model_selector_callback=on_model_change,
            embedding_selector_callback=on_embedding_change,
            new_chat_callback=on_new_chat,
            refresh_models_callback=on_refresh_models,
            is_generating=bool(SessionManager.get("is_generating_answer", False)),
            current_file_name=SessionManager.get("last_uploaded_file_name"),
            available_models=available_models,
        )

    from ui.ui import render_main_content

    render_main_content()


def _handle_pending_tasks() -> None:
    from core.session import SessionManager

    current_sid = SessionManager.get_session_id()
    is_building = bool(SessionManager.get("is_building_rag", False, current_sid))
    needs_rerun = False

    if (
        SessionManager.get("new_file_uploaded", False, current_sid)
        and not is_building
        and not SessionManager.get("is_generating_answer", False, current_sid)
    ):
        SessionManager.set("new_file_uploaded", False, current_sid)
        SessionManager.set("is_building_rag", True, current_sid)

        current_file_path = SessionManager.get("pdf_file_path", None, current_sid)
        current_file_name = SessionManager.get(
            "last_uploaded_file_name", None, current_sid
        )
        current_embedding_model = (
            SessionManager.get("last_selected_embedding_model", None, current_sid)
            or DEFAULT_EMBEDDING_MODEL
        )

        # [UX] 리셋/진행 메시지는 on_file_upload가 렌더 전에 이미 수행했으므로
        # 여기서는 빌드 시작만 한다. (st.rerun 제거: 갱신은 2초 폴링 fragment와
        # run_in_background_worker의 완료 시 rerun이 담당)
        from common.utils import run_in_background_worker

        run_in_background_worker(
            _bg_rebuild_task(
                current_sid,
                current_file_path,
                current_file_name,
                current_embedding_model,
            ),
            current_sid,
        )

    elif (
        SessionManager.get("needs_rag_rebuild", False, current_sid)
        and not is_building
        and not SessionManager.get("is_generating_answer", False, current_sid)
    ):
        SessionManager.set("needs_rag_rebuild", False, current_sid)
        SessionManager.set("is_building_rag", True, current_sid)

        current_file_path = SessionManager.get("pdf_file_path", None, current_sid)
        current_file_name = SessionManager.get(
            "last_uploaded_file_name", None, current_sid
        )
        current_embedding_model = SessionManager.get(
            "last_selected_embedding_model", None, current_sid
        )

        from common.utils import run_in_background_worker

        run_in_background_worker(
            _bg_rebuild_task(
                current_sid,
                current_file_path,
                current_file_name,
                current_embedding_model,
            ),
            current_sid,
        )
        needs_rerun = True

    elif SessionManager.get("needs_qa_chain_update", False, current_sid):
        SessionManager.set("needs_qa_chain_update", False, current_sid)
        _update_qa_chain(current_sid)
        needs_rerun = True

    if needs_rerun:
        st.rerun()


_SPLASH_HTML = """
<div style="text-align: center; padding: 80px 0;">
    <div style="font-size: 2rem; font-weight: 700; color: var(--text-color);">
        GraphRAG-Ollama
    </div>
    <div style="font-size: 1rem; color: var(--primary-color); opacity: 0.8; margin-top: 8px;">
        Local RAG · PDF Chat
    </div>
</div>
"""


def _ensure_ui_globals() -> None:
    """Pass-1-safe UI bootstrap: session init + bridge sync."""
    from core.session import SessionManager
    from ui.bridge import UIBridge

    SessionManager.init_session()
    UIBridge.sync_session()


def main() -> None:
    from core.session import SessionManager
    from ui.ui import inject_custom_css

    inject_custom_css()  # light import (ui.ui only); CSS lands in the first frame
    _ensure_ui_globals()  # heavy session init AFTER css is already streamed

    # 부트 스플래시: 1차 패스에서 즉시 시각 피드백 후 1회 rerun (2차 패스에서 본 UI 렌더)
    if not st.session_state.get("_bootstrapped"):
        st.session_state._bootstrapped = True
        st.markdown(_SPLASH_HTML, unsafe_allow_html=True)
        st.status("🔄 앱 초기화 중...", state="running")
        st.rerun()

    if "available_models_list" not in st.session_state:
        st.session_state.available_models_list = _load_available_models()

    if SessionManager.get("pdf_file_path") and not SessionManager.get("pdf_processed"):
        SessionManager.set("is_generating_answer", False)

    available_models = st.session_state.available_models_list
    _render_app_layout(available_models=available_models)
    _handle_pending_tasks()

    if SessionManager.get("is_first_run"):
        SessionManager.set("is_first_run", False)


if __name__ == "__main__":
    main()
