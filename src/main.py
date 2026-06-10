"""
RAG Chatbot 애플리케이션의 메인 진입점 - Native Streaming Architecture
"""

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

from common.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_MODEL,
)
from common.constants import FilePathConstants, StringConstants
from common.logging_config import setup_logging
from common.utils import safe_cache_resource, sync_run

st.set_page_config(
    page_title=StringConstants.PAGE_TITLE,
    layout=cast(Literal["centered", "wide"], StringConstants.LAYOUT),
    initial_sidebar_state="expanded",
)

logger = setup_logging(log_level="DEBUG", log_file=Path("logs/app.log"))

MAX_FILE_SIZE_MB = StringConstants.MAX_FILE_SIZE_MB

if "current_page" not in st.session_state:
    st.session_state.current_page = 1
if "pdf_target_page" not in st.session_state:
    st.session_state.pdf_target_page = None
if "is_generating_answer" not in st.session_state:
    st.session_state.is_generating_answer = False
if "sidebar_auto_collapsed" not in st.session_state:
    st.session_state.sidebar_auto_collapsed = False


def _check_windows_integrity():
    try:
        from core.session import SessionManager

        SessionManager.cleanup_expired_sessions(max_idle_seconds=3600)
    except Exception as e:
        logger.error(f"[SYSTEM] [CLEANUP] 세션 정리 중 오류: {e}")


@st.cache_resource
def _start_global_background_worker():
    def maintenance_loop():
        while True:
            with contextlib.suppress(Exception):
                _check_windows_integrity()
            time.sleep(3600)

    thread = threading.Thread(target=maintenance_loop, daemon=True)
    thread.start()
    return thread


@st.cache_data(ttl=300)
def _get_available_models_cached():
    from core.model_loader import get_available_models

    return get_available_models()


@safe_cache_resource(show_spinner=False)
def _init_temp_directory():
    temp_path = Path(FilePathConstants.TEMP_DIR).absolute()
    temp_path.mkdir(parents=True, exist_ok=True)
    return str(temp_path)


def _cleanup_current_file():
    from core.session import SessionManager

    try:
        path = SessionManager.get("pdf_file_path", create=False)
        if path:
            SessionManager.safe_remove_file(path)
    except Exception:
        pass


@st.cache_resource
def _register_cleanup_handlers():
    atexit.register(_cleanup_current_file)
    return True


_init_temp_directory()
_start_global_background_worker()
_register_cleanup_handlers()


async def _bg_rebuild_task(
    session_id: str, file_path: str, file_name: str, embedder_name: str
):
    from core.model_loader import ModelManager
    from core.rag_core import RAGSystem
    from core.session import SessionManager

    SessionManager.set_session_id(session_id)
    SessionManager.set("rebuild_done", False)
    SessionManager.set("rebuild_error", None)
    SessionManager.set("rebuild_status", f"'{file_name}' 분석 중...")

    try:
        embedder = await ModelManager.get_embedder(embedder_name)
        rag_sys = RAGSystem(session_id=session_id)

        success_message, cache_used = await rag_sys.build_pipeline(
            file_path=file_path, file_name=file_name, embedder=embedder
        )

        SessionManager.set("pdf_processed", True)
        SessionManager.add_status_log(f"✅ {success_message}")
        SessionManager.add_message("system", success_message)
    except Exception as e:
        logger.error(f"Background RAG rebuild error: {e}", exc_info=True)
        error_msg = f"문서 처리 중 오류가 발생했습니다: {str(e)}"
        SessionManager.set("rebuild_error", error_msg)
        SessionManager.set("pdf_processing_error", error_msg)
        SessionManager.set("pdf_processed", True)
        SessionManager.add_message("system", f"❌ {error_msg}")
    finally:
        SessionManager.set("rebuild_done", True)
        SessionManager.set("is_building_rag", False)


def _update_qa_chain(session_id: str | None = None) -> None:
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
        st.error("❌ 올바른 PDF 파일이 아닙니다.")
        return

    if uploaded_file.name != SessionManager.get("last_uploaded_file_name"):
        st.session_state.sidebar_auto_collapsed = False  # Reset flag for new file
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
        from ui.components.viewer import render_pdf_column

        render_pdf_column()
    with col_chat:
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
