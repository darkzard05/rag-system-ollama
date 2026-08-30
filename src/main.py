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

# [Path Bootstrap] 저장소 루트의 src/ 를 sys.path 에 추가하여
# `from common`/`from core` 식의 bare import 가 해석되도록 한다.
# (프로젝트 스크립트들이 사용하는 관례: scripts/test_app.py 등)
# 이 부트스트랩이 없으면 streamlit run src/main.py 가 저장소 루트에서
# 실행될 때 `No module named 'src.common'` 으로 실패한다.
import sys
from pathlib import Path

_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

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
from typing import Any, Literal, cast

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
# LOG_LEVEL 환경변수로 DEBUG 전환 가능 (기본 INFO). JSON 파싱 실패 등 원시
# 출력 추적이 필요할 때 LOG_LEVEL=DEBUG 로 서버를 기동한다.
@st.cache_resource
def _init_logging() -> logging.Logger:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    return setup_logging(log_level=log_level, log_file=FilePathConstants.LOG_FILE)


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


@st.cache_data(ttl=300, show_spinner=False)
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


def _warm_available_models() -> None:
    """RC-C: 프로세스 부트 시 Ollama 모델 목록을 비동기 프리워밍합니다.

    _get_available_models_cached(ttl=300)는 app 런타임 캐시이므로 import 시점
    호출은 부팅 미리채우기에 쓰이지 않는다. 대신 데몬 스레드로 최대 1회만
    조회하여 캐시를 데우고, 첫 사용자 렌더가 ~5s 블로킹(list 호출)되는 것을
    막는다. Ollama 미연결 시 예외는 무시 — 첫 렌더의 기존 폴백/스피너가 보완.
    """
    try:
        _get_available_models_cached()
    except Exception:  # noqa: BLE001 - best-effort warmup; first render handles failure
        logger.debug("[SYSTEM] model-list warmup skipped (Ollama not ready)")


# 프로세스 시작 직후(import 직후) 백그라운드 웜업 — UI 첫 렌더 블로킹 제거.
threading.Thread(
    target=_warm_available_models, name="model-list-warmup", daemon=True
).start()


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

# F1: attach the Streamlit UI-sync adapter so SessionManager can mirror state
# into st.session_state without core importing streamlit.
from core.session import SessionManager
from ui.session_sync import StreamlitSessionSync

SessionManager.set_ui_sync(StreamlitSessionSync())


async def _warmup_models() -> None:
    """[WARMUP] 시작 시 LLM+임베더 1회 프리웜 — 첫 쿼리 TTFT 제거.

    AsyncWorker 전용 루프에서 비차단으로 실행된다. Ollama 미연결 등 실패 시에도
    UI 시작은 계속되어야 하므로 호출부에서 비치명적으로 감싼다.
    """
    from core.model_loader import _warmup_models as _core_warmup

    await _core_warmup()


# [WARMUP] 부팅 시 1회 프리웜을 백그라운드 워커에 제출(비차단, 비치명적).
# Streamlit은 스크립트 rerun 시 모듈 최상위 코드를 재실행하므로, 가드 없으면
# 매 rerun마다 프리웜이 중복 제출되어 LLM astream("warmup") 이多次 실행된다.
# session_state 플래그로 1회만 제출되도록 보장한다.
if not st.session_state.get("_warmup_submitted", False):
    try:
        _async_worker.submit(_warmup_models())
        st.session_state["_warmup_submitted"] = True
    except Exception as e:
        logger.warning(f"[WARMUP] 모델 프리웜 실패 — 첫 쿼리에서 로드됨: {e}")


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
        "rebuild_status", f"Analyzing '{file_name}'...", session_id=session_id
    )

    # 타임라인에 분석 시작 메시지 추가 (하나의 메시지를 업데이트)
    # on_file_upload가 만든 자리표시자(msg_id=build_{session_id})를 재사용하여
    # 빈 대화 깜빡임 없이 진행 상황이 그 위에 업데이트된다.
    build_msg_id = f"build_{session_id}"
    SessionManager.set("build_msg_id", build_msg_id, session_id=session_id)
    SessionManager.add_message(
        "system",
        f"Analysis started for '{file_name}'",
        msg_type="build_progress",
        msg_id=build_msg_id,
        progress=0,
        status="Preparing analysis...",
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
                "Document analysis cancelled.", session_id=session_id
            )
            SessionManager.add_message(
                "system",
                "Document analysis cancelled",
                msg_type="build_error",
                msg_id=build_msg_id,
                error="cancelled by the user",
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
                f"Analyzing '{file_name}'...",
                msg_type="build_progress",
                msg_id=build_msg_id,
                progress=pct,
                status=msg or f"Progress {pct}%",
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
                "Document analysis cancelled.", session_id=session_id
            )
            SessionManager.add_message(
                "system",
                "Document analysis cancelled",
                msg_type="build_error",
                msg_id=build_msg_id,
                error="cancelled by the user",
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
        SessionManager.add_status_log(success_message, session_id=session_id)
        # 타임라인 진행 메시지 완료 처리 (동일 msg_id)
        SessionManager.add_message(
            "system",
            success_message,
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
            "Document analysis cancelled",
            msg_type="build_error",
            msg_id=build_msg_id,
            error="async task cancelled",
            session_id=session_id,
        )
    except Exception as e:
        logger.error(f"Background RAG rebuild error: {e}", exc_info=True)
        error_msg = f"An error occurred while processing the document: {e}"
        SessionManager.set("rebuild_error", error_msg, session_id=session_id)
        SessionManager.set("pdf_processing_error", error_msg, session_id=session_id)
        SessionManager.set("rebuild_progress", 0, session_id=session_id)
        SessionManager.set("pdf_processed", False, session_id=session_id)
        SessionManager.add_message(
            "system",
            "Error during document analysis",
            msg_type="build_error",
            msg_id=build_msg_id,
            error=error_msg,
            session_id=session_id,
        )
        SessionManager.add_message(
            "assistant",
            error_msg,
            msg_type="build_error",
            session_id=session_id,
        )
    finally:
        SessionManager.set("rebuild_done", True, session_id=session_id)
        SessionManager.set("is_building_rag", False, session_id=session_id)


async def _bg_update_qa_chain(session_id: str) -> None:
    """
    [Background Task] 문서 인덱싱은 유지한 채 LLM(QA Chain)만 백그라운드로 교체합니다.

    `_bg_rebuild_task`와 동일하게 run_in_background_worker(common.utils)가 전용
    AsyncWorker 이벤트 루프에서 실행한다. load_llm은 동기 호출이므로
    `asyncio.to_thread`로 스레드 풀에 오프로드한다 — AsyncWorker는 단일 루프이며
    스트림 소비자(ui/components/streaming.py)와 공유되므로 루프에서 직접 실행하면
    UI 스트리밍까지 함께 블로킹된다.
    """
    from core.model_loader import load_llm
    from core.session import SessionManager

    SessionManager.set_session_id(session_id)
    SessionManager.set("is_swapping_model", True, session_id=session_id)
    try:
        SessionManager.add_status_log(
            "Switching inference model...", session_id=session_id
        )
        selected_model = SessionManager.get(
            "last_selected_model", session_id=session_id
        )
        model_name = str(selected_model or DEFAULT_OLLAMA_MODEL)
        llm = await asyncio.to_thread(load_llm, model_name)
        SessionManager.set("llm", llm, session_id=session_id)
        SessionManager.add_status_log("Inference model switched", session_id=session_id)
    except Exception as e:
        error_msg = f"Failed to update the QA chain: {e}"
        logger.error(f"QA 업데이트 실패: {e}", exc_info=True)
        SessionManager.add_status_log(error_msg, session_id=session_id)
        SessionManager.add_message(
            "assistant",
            error_msg,
            msg_type="build_error",
            session_id=session_id,
        )
    finally:
        SessionManager.set("rag_build_complete_flag", True, session_id=session_id)
        SessionManager.set("is_swapping_model", False, session_id=session_id)


def _update_qa_chain(session_id: str | None = None) -> None:
    """
    문서 인덱싱은 유지한 채 LLM(QA Chain)만 교체합니다.
    """
    from core.session import SessionManager

    sid = session_id or SessionManager.get_session_id()
    selected_model = SessionManager.get("last_selected_model", session_id=sid)
    try:
        SessionManager.add_status_log("Switching inference model...", session_id=sid)
        from core.model_loader import load_llm

        model_name = str(selected_model or DEFAULT_OLLAMA_MODEL)
        llm = load_llm(model_name)
        SessionManager.set("llm", llm, session_id=sid)
        SessionManager.add_status_log("Inference model switched", session_id=sid)
    except Exception as e:
        error_msg = f"Failed to update the QA chain: {e}"
        logger.error(f"QA 업데이트 실패: {e}", exc_info=True)
        SessionManager.add_status_log(error_msg, session_id=sid)
        SessionManager.add_message(
            "assistant",
            error_msg,
            msg_type="build_error",
            session_id=sid,
        )
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
    SessionManager.add_message(
        "assistant",
        error_msg,
        msg_type="build_error",
        session_id=sid,
    )


def on_file_upload() -> None:
    from core.document_processor import compute_file_hash
    from core.session import SessionManager
    from infra.notification_system import SystemNotifier

    uploaded_file = st.session_state.get("pdf_uploader")
    if uploaded_file is None or not hasattr(uploaded_file, "type"):
        return

    if uploaded_file.type != "application/pdf":
        _post_upload_error("Invalid PDF file. Please upload a file in PDF format.")
        return

    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        _post_upload_error(
            f"File is too large ({file_size_mb:.2f} MB). Please upload a file no larger than {MAX_FILE_SIZE_MB} MB."
        )
        return

    # [멱등 가드] Streamlit rerun/더블 트리거로 on_change 콜백이 동일 파일에
    # 대해 중복 발동되는 것을 차단한다. 해시 계산 전에 세션에 처리 중 플래그를
    # 선점 세트하여, 재진입 시점에 이미 처리 중이면 즉시 종료한다.
    upload_guard_key = "file_upload_in_progress"
    if SessionManager.get(upload_guard_key, False):
        return
    SessionManager.set(upload_guard_key, True)
    try:
        _process_uploaded_file(uploaded_file)
    finally:
        SessionManager.set(upload_guard_key, False)


def _process_uploaded_file(uploaded_file) -> None:
    """실제 업로드 처리 로직. on_file_upload 가 멱등 가드 후 호출한다."""
    from core.document_processor import compute_file_hash
    from core.session import SessionManager
    from infra.notification_system import SystemNotifier

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
        import pymupdf as fitz  # lazy import: 검증이 필요한 시점에만 로드

        if not file_bytes:
            raise ValueError("PDF file is empty.")
        with fitz.open(stream=file_bytes, filetype="pdf") as upload_doc:
            page_count = len(upload_doc)
        if page_count < 1:
            raise ValueError("PDF file has no pages.")
    except Exception as e:
        logger.warning(f"손상되었거나 읽을 수 없는 PDF 업로드 차단: {e}")
        _post_upload_error(
            "The PDF file is corrupted or unreadable. Please upload a valid PDF file."
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
                f"Analysis started for '{uploaded_file.name}'",
                msg_type="build_progress",
                msg_id=f"build_{sid}",
                progress=0,
                status="Preparing analysis...",
                cancelable=True,
                logs=[],
                session_id=sid,
            )
            SessionManager.set("new_file_uploaded", True)
            # 선점: on_change 자동 rerun의 첫 페인트 프레임이 빌드 메시지와
            # 동일한 정착 상태를 보도록 is_building_rag를 미리 True로 세팅.
            # reset_for_new_file은 이 플래그를 건드리지 않음. 디스패치 가드는
            # _handle_pending_tasks에서 new_file_uploaded 단독으로 판단한다.
            SessionManager.set("is_building_rag", True)
            SystemNotifier.success(f"Document uploaded: {uploaded_file.name}")
        except Exception as e:
            SystemNotifier.error("Error while saving the file", details=str(e))
    else:
        SystemNotifier.success(f"'{uploaded_file.name}' was already uploaded.")


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
            is_swapping_model=bool(SessionManager.get("is_swapping_model", False)),
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

    def _safe_dispatch(kind: str, coro: Any) -> None:
        """백그라운드 디스패치를 감싸 실패 시 플래그가 영구 잔류되지 않게 한다.

        ``run_in_background_worker`` 가 submit 에서 실패하면 ``_on_complete``
        가 영원히 호출되지 않아 ``is_building_rag``/``needs_*`` 플래그가
        굳어 입력창이 비활성화된다(INT-입력동결). 디스패치 직후 죽으면
        여기서 플래그를 롤백한다.
        """
        from common.utils import run_in_background_worker

        try:
            run_in_background_worker(coro, current_sid)
        except Exception as exc:  # noqa: BLE001 - 디스패치 실패는 복구해야 함
            logger.error("[MAIN] 백그라운드 디스패치 실패 (%s): %s", kind, exc)
            if kind == "rebuild":
                SessionManager.set("needs_rag_rebuild", True, current_sid)
            elif kind == "qa_update":
                SessionManager.set("needs_qa_chain_update", True, current_sid)
            SessionManager.set("is_building_rag", False, current_sid)
            SessionManager.set(
                "pdf_processing_error",
                f"Background task dispatch failed: {exc}",
                current_sid,
            )

    if SessionManager.get(
        "new_file_uploaded", False, current_sid
    ) and not SessionManager.get("is_generating_answer", False, current_sid):
        SessionManager.set("new_file_uploaded", False, current_sid)
        # is_building_rag는 업로드 콜백에서 이미 True로 선점됨. 가드를
        # new_file_uploaded 단독으로 풀어 디스패치가 차단되지 않게 한다.

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
        _safe_dispatch(
            "rebuild",
            _bg_rebuild_task(
                current_sid,
                current_file_path,
                current_file_name,
                current_embedding_model,
            ),
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

        _safe_dispatch(
            "rebuild",
            _bg_rebuild_task(
                current_sid,
                current_file_path,
                current_file_name,
                current_embedding_model,
            ),
        )
        needs_rerun = True

    elif SessionManager.get("needs_qa_chain_update", False, current_sid):
        SessionManager.set("needs_qa_chain_update", False, current_sid)
        # [INT-3] 이미 교체 진행 중이면 중복 디스패치 금지 (is_swapping_model 가드)
        if not SessionManager.get("is_swapping_model", False, current_sid):
            SessionManager.set("is_swapping_model", True, current_sid)

            _safe_dispatch("qa_update", _bg_update_qa_chain(current_sid))
        needs_rerun = True

    if needs_rerun:
        st.rerun()


def _ensure_ui_globals() -> None:
    """Pass-1-safe UI bootstrap: session init + bridge sync."""
    from core.session import SessionManager
    from ui.bridge import UIBridge

    SessionManager.init_session()
    UIBridge.sync_session()


def main() -> None:
    from core.session import SessionManager
    from ui.ui import inject_custom_css

    _t_main = time.perf_counter()
    inject_custom_css()  # light import (ui.ui only); CSS lands in the first frame
    _ensure_ui_globals()  # heavy session init AFTER css is already streamed
    logger.debug("[PERF] main(): css+globals took %.3fs", time.perf_counter() - _t_main)

    # 부트 스플래시 제거: 스플래시는 실제 지연(모듈 스코프 웜업·첫 쿼리 로드)을
    # 커버하지 못하는 가짜 로더였다. 유일한 실제 블로킹(_load_available_models,
    # Ollama list 최대 5s)에만 정직한 스피너를 표시한다.
    if "available_models_list" not in st.session_state:
        _t_models = time.perf_counter()
        with st.spinner("Loading available models…"):
            st.session_state.available_models_list = _load_available_models()
        logger.debug(
            "[PERF] main(): _load_available_models took %.3fs",
            time.perf_counter() - _t_models,
        )

    if SessionManager.get("pdf_file_path") and not SessionManager.get("pdf_processed"):
        SessionManager.set("is_generating_answer", False)

    available_models = st.session_state.available_models_list
    _t_layout = time.perf_counter()
    _render_app_layout(available_models=available_models)
    logger.debug(
        "[PERF] main(): _render_app_layout(sidebar) took %.3fs",
        time.perf_counter() - _t_layout,
    )
    _t_pending = time.perf_counter()
    _handle_pending_tasks()
    logger.debug(
        "[PERF] main(): _handle_pending_tasks took %.3fs",
        time.perf_counter() - _t_pending,
    )
    logger.debug("[PERF] main(): TOTAL rerun %.3fs", time.perf_counter() - _t_main)

    if SessionManager.get("is_first_run"):
        SessionManager.set("is_first_run", False)


if __name__ == "__main__":
    main()
