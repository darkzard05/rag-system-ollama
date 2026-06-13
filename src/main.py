# src/main.py
# Streamlit 기반 RAG 챗봇의 메인 진입점 및 전체 오케스트레이션을 담당하는 파일
"""
RAG Chatbot 애플리케이션의 메인 진입점 파일입니다.
Streamlit 프레임워크를 기반으로 UI를 구성하고 세션 상태를 관리합니다.
"""

# [Lazy Import용] 런타임에 필요한 모듈들
import atexit
import contextlib
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Literal, cast

import nest_asyncio
import streamlit as st

from common.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_MODEL,
)
from common.constants import FilePathConstants, StringConstants
from common.logging_config import setup_logging
from common.utils import safe_cache_resource, sync_run

# 1. Streamlit 페이지 설정 (최우선 실행 - 가이드라인 준수)
st.set_page_config(
    page_title=StringConstants.PAGE_TITLE,
    layout=cast(Literal["centered", "wide"], StringConstants.LAYOUT),
    initial_sidebar_state="expanded",
)

# 2. 로깅 설정 (최상단)
logger = setup_logging(log_level="DEBUG", log_file=Path("logs/app.log"))

MAX_FILE_SIZE_MB = StringConstants.MAX_FILE_SIZE_MB

# 비동기 패치 적용 (Streamlit의 asyncio 루프 충돌 방지)
nest_asyncio.apply()

if "current_page" not in st.session_state:
    st.session_state.current_page = 1
if "pdf_target_page" not in st.session_state:
    st.session_state.pdf_target_page = None
if "is_generating_answer" not in st.session_state:
    st.session_state.is_generating_answer = False
if "sidebar_auto_collapsed" not in st.session_state:
    st.session_state.sidebar_auto_collapsed = False


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
        import sys

        if "torch" not in sys.modules:
            import torch
            _ = torch.tensor([1.0])

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


@safe_cache_resource(show_spinner=False)
def _init_temp_directory():
    """임시 디렉토리를 초기화합니다."""
    temp_path = Path(FilePathConstants.TEMP_DIR).absolute()
    temp_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"[SYSTEM] [INIT] 임시 디렉토리 준비 완료: {temp_path}")
    return str(temp_path)


def _cleanup_current_file():
    """현재 세션에서 사용 중인 임시 파일을 삭제합니다. (종료 핸들러용)"""
    from core.session import SessionManager

    try:
        path = SessionManager.get("pdf_file_path", create=False)
        if path:
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

    try:
        embedder = await ModelManager.get_embedder(embedder_name)
        rag_sys = RAGSystem(session_id=session_id)

        success_message, cache_used = await rag_sys.build_pipeline(
            file_path=file_path, file_name=file_name, embedder=embedder
        )

        SessionManager.set("pdf_processed", True, session_id=session_id)
        SessionManager.add_status_log(f"✅ {success_message}", session_id=session_id)
        SessionManager.add_message("system", success_message, session_id=session_id)
    except Exception as e:
        logger.error(f"Background RAG rebuild error: {e}", exc_info=True)
        error_msg = f"문서 처리 중 오류가 발생했습니다: {str(e)}"
        SessionManager.set("rebuild_error", error_msg, session_id=session_id)
        SessionManager.set("pdf_processing_error", error_msg, session_id=session_id)
        SessionManager.set("pdf_processed", True, session_id=session_id)
        SessionManager.add_message("system", f"❌ {error_msg}", session_id=session_id)
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
        logger.error(f"QA 업데이트 실패: {e}", exc_info=True)
    finally:
        SessionManager.set("rag_build_complete_flag", True, session_id=sid)


def on_file_upload() -> None:
    from core.session import SessionManager
    from infra.notification_system import SystemNotifier

    uploaded_file = st.session_state.get("pdf_uploader")
    if uploaded_file is None or not hasattr(uploaded_file, "type"):
        return

    if uploaded_file.type != "application/pdf":
        st.error("❌ 올바른 PDF 파일이 아닙니다. PDF 형식의 파일을 업로드해주세요.")
        return

    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(
            f"❌ 파일 크기가 너무 큽니다 ({file_size_mb:.2f} MB). {MAX_FILE_SIZE_MB}MB 이하의 파일을 업로드해주세요."
        )
        return

    if uploaded_file.name != SessionManager.get("last_uploaded_file_name"):
        st.session_state.sidebar_auto_collapsed = False
        SessionManager.set("current_page", 1)
        old_path = SessionManager.get("pdf_file_path")
        if old_path:
            SessionManager.safe_remove_file(old_path)

        SessionManager.set("last_uploaded_file_name", uploaded_file.name)

        try:
            temp_dir = os.path.abspath(FilePathConstants.TEMP_DIR)
            sid = SessionManager.get_session_id()
            tmp_path = os.path.join(temp_dir, f"upload_{sid}_{int(time.time())}.pdf")

            with open(tmp_path, "wb") as f:
                shutil.copyfileobj(uploaded_file, f)

            SessionManager.set("pdf_file_path", tmp_path)
            SessionManager.set("new_file_uploaded", True)
            SystemNotifier.success(f"문서 업로드 완료: {uploaded_file.name}")
        except Exception as e:
            SystemNotifier.error("파일 저장 중 오류 발생", details=str(e))


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


def _render_app_layout(available_models: list[str] | None = None) -> None:
    from core.session import SessionManager
    from ui.components.sidebar import render_settings_content
    from streamlit_js_eval import streamlit_js_eval

    # Get browser height dynamically
    viewport_height = streamlit_js_eval(js_expressions="window.innerHeight", key="viewport_height")
    # Calculate target height (header/padding offset: ~150px)
    # Default to 800 if not yet detected
    target_height = (viewport_height - 150) if viewport_height else 800

    with st.sidebar:
        render_settings_content(
            file_uploader_callback=on_file_upload,
            model_selector_callback=on_model_change,
            embedding_selector_callback=on_embedding_change,
            is_generating=bool(SessionManager.get("is_generating_answer", False)),
            current_file_name=SessionManager.get("last_uploaded_file_name"),
            available_models=available_models,
        )

    col_pdf, col_chat = st.columns([1, 1], gap="medium")
    with col_pdf:
        with st.container(height=target_height, border=False):
            from ui.components.viewer import render_pdf_column
            render_pdf_column()
    with col_chat:
        with st.container(height=target_height, border=False):
            from ui.ui import render_left_column
            render_left_column()


def _handle_pending_tasks() -> None:
    from core.session import SessionManager

    current_sid = SessionManager.get_session_id()
    is_building = bool(SessionManager.get("is_building_rag", False, current_sid))

    if SessionManager.get("new_file_uploaded", False, current_sid) and not is_building:
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

        SessionManager.reset_for_new_file(current_sid)
        SessionManager.set("pdf_file_path", current_file_path, current_sid)
        SessionManager.set("last_uploaded_file_name", current_file_name, current_sid)
        SessionManager.set(
            "last_selected_embedding_model", current_embedding_model, current_sid
        )
        SessionManager.set("is_building_rag", True, current_sid)

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
        st.rerun()

    elif (
        SessionManager.get("needs_rag_rebuild", False, current_sid) and not is_building
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
        st.rerun()

    elif SessionManager.get("needs_qa_chain_update", False, current_sid):
        SessionManager.set("needs_qa_chain_update", False, current_sid)
        _update_qa_chain(current_sid)
        st.rerun()


def main() -> None:
    from core.session import SessionManager
    from ui.ui import inject_custom_css

    SessionManager.init_session()

    # UI 렌더 단계에서 Streamlit 상태 동기화 (스레드 안전)
    SessionManager.sync_to_streamlit()

    if "available_models_list" not in st.session_state:
        with st.spinner("시스템 초기화 중..."):
            fetched_models = _get_available_models_cached()
            st.session_state.available_models_list = (
                fetched_models if fetched_models else [DEFAULT_OLLAMA_MODEL]
            )
        st.rerun()

    is_expanded = bool(
        SessionManager.get("pdf_file_path")
    ) and not st.session_state.get("sidebar_collapsed", False)
    inject_custom_css(is_expanded=is_expanded)

    if SessionManager.get("pdf_file_path") and not SessionManager.get("pdf_processed"):
        SessionManager.set("is_generating_answer", False)

    available_models = st.session_state.available_models_list
    _render_app_layout(available_models=available_models)
    _handle_pending_tasks()

    if SessionManager.get("is_first_run"):
        SessionManager.set("is_first_run", False)


if __name__ == "__main__":
    main()