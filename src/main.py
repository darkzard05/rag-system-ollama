"""
RAG Chatbot 애플리케이션의 메인 진입점 파일입니다.
Streamlit 프레임워크를 기반으로 UI를 구성하고 세션 상태를 관리합니다.
"""

import logging
import tempfile
import os
from typing import Any, Dict
from pathlib import Path

import nest_asyncio
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

# 로깅 설정 (최상단)
from common.logging_config import setup_logging, get_logger
logger = setup_logging(
    log_level="INFO",
    log_file=Path("logs/app.log")
)

from common.config import AVAILABLE_EMBEDDING_MODELS, DEFAULT_OLLAMA_MODEL
from common.constants import StringConstants, FilePathConstants
# [Lazy Import] 무거운 코어 모듈 임포트 제거 (함수 내부로 이동)
from core.session import SessionManager
from ui.ui import render_left_column, render_pdf_viewer, render_sidebar, _render_status_box, inject_custom_css

# 상수 정의
PAGE_TITLE = StringConstants.PAGE_TITLE
LAYOUT = StringConstants.LAYOUT
MAX_FILE_SIZE_MB = StringConstants.MAX_FILE_SIZE_MB

# 비동기 패치 적용 (최상단 실행)
nest_asyncio.apply()

# Streamlit 페이지 설정
st.set_page_config(page_title=PAGE_TITLE, layout=LAYOUT)


import threading
import atexit
import shutil

@st.cache_resource
def _init_temp_directory():
    """임시 디렉토리를 초기화하고 이전의 잔해를 제거합니다. (앱 시작 시 1회 실행)"""
    temp_path = Path(FilePathConstants.TEMP_DIR).absolute()
    try:
        if temp_path.exists():
            # 안전을 위해 폴더 내부 파일만 삭제
            for file in temp_path.glob("*.pdf"):
                try:
                    os.remove(file)
                except: pass
            logger.info(f"[System] [Cleanup] 임시 디렉토리 초기화 완료: {temp_path}")
        else:
            temp_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"[System] [Cleanup] 임시 디렉토리 생성 완료: {temp_path}")
    except Exception as e:
        logger.warning(f"임시 디렉토리 초기화 실패: {e}")
    return True

# 앱 시작 시 초기화 수행 (캐싱으로 인해 최초 1회만 작동)
_init_temp_directory()

def _cleanup_current_file():
    """현재 세션에서 사용 중인 임시 파일을 삭제합니다. (종료 핸들러용)"""
    # Streamlit 세션 상태를 직접 접근하기 어려우므로 SessionManager는 thread-safe하게 설계됨
    try:
        path = SessionManager.get("pdf_file_path")
        if path and os.path.exists(path):
            os.remove(path)
            # logger는 이미 닫혔을 수 있으므로 print 사용
            print(f"[System] Cleanup: Deleted temp file {path}")
    except: pass

# 앱 시작 시 초기화 수행
_init_temp_directory()
# 프로세스 종료 시 핸들러 등록
atexit.register(_cleanup_current_file)

def _ensure_models_are_loaded(status_container: DeltaGenerator) -> bool:
    """
    선택된 LLM 및 임베딩 모델을 순차적으로 로드하여 안정성을 확보합니다.
    """
    # [Lazy Import]
    from core.model_loader import load_embedding_model, load_llm

    selected_model = SessionManager.get("last_selected_model")
    selected_embedding = SessionManager.get("last_selected_embedding_model")


    if not selected_model:
        st.warning("⚠️ LLM 모델이 선택되지 않았습니다.")
        return False

    if not selected_embedding:
        if AVAILABLE_EMBEDDING_MODELS:
            selected_embedding = AVAILABLE_EMBEDDING_MODELS[0]
            SessionManager.set("last_selected_embedding_model", selected_embedding)
        else:
            st.error("❌ 사용 가능한 임베딩 모델 설정이 없습니다.")
            return False

    try:
        status_placeholder = SessionManager.get("status_placeholder")
        def force_sync():
            if status_placeholder:
                _render_status_box(status_placeholder)

        current_llm = SessionManager.get("llm")
        current_embedder = SessionManager.get("embedder")
        
        # 1. LLM 로드
        if not current_llm or getattr(current_llm, "model", None) != selected_model:
            SessionManager.add_status_log(f"LLM 로딩 중: {selected_model}")
            force_sync()
            llm = load_llm(selected_model)
            SessionManager.set("llm", llm)
            SessionManager.replace_last_status_log(f"✅ LLM 로드 완료")
            st.toast(f"LLM 로드 완료: {selected_model}", icon="✅")
            force_sync()

        # 2. 임베딩 모델 로드
        if not current_embedder or getattr(current_embedder, "model_name", None) != selected_embedding:
            SessionManager.add_status_log(f"임베딩 로딩 중: {selected_embedding}")
            force_sync()
            embedder = load_embedding_model(selected_embedding)
            SessionManager.set("embedder", embedder)
            SessionManager.replace_last_status_log(f"✅ 임베딩 로드 완료")
            st.toast(f"임베딩 모델 로드 완료: {selected_embedding}", icon="✅")
            force_sync()

        return True

    except Exception as e:
        logger.error(f"모델 로드 중 치명적 오류 발생: {e}", exc_info=True)
        status_container.error(f"❌ 모델 로드 실패: {e}")
        return False


def _rebuild_rag_system(status_container: DeltaGenerator) -> None:
    """
    업로드된 파일과 선택된 모델을 사용하여 RAG 파이프라인을 재구축합니다.
    """
    file_name = SessionManager.get("last_uploaded_file_name")
    file_path = SessionManager.get("pdf_file_path")

    if not file_name or not file_path:
        return

    # [중복 실행 방지]
    if (SessionManager.get("pdf_processed") 
        and not SessionManager.get("pdf_processing_error") 
        and SessionManager.get("vector_store") is not None):
        return

    try:
        if not _ensure_models_are_loaded(status_container):
            return

        embedder = SessionManager.get("embedder")
        
        # 실시간 상태 박스 업데이트를 위한 콜백 정의
        status_placeholder = SessionManager.get("status_placeholder")
        def sync_ui():
            if status_placeholder:
                _render_status_box(status_placeholder)

        # [Lazy Import]
        from core.rag_core import build_rag_pipeline

        # RAG 파이프라인 빌드 (내부에서 상세 로그 기록 및 UI 동기화)
        success_message, cache_used = build_rag_pipeline(
            uploaded_file_name=file_name,
            file_path=file_path,
            embedder=embedder,
            on_progress=sync_ui
        )

        SessionManager.add_message("assistant", success_message)

    except Exception as e:
        logger.error(f"RAG 빌드 실패: {e}", exc_info=True)
        error_msg = f"문서 처리 중 오류가 발생했습니다: {str(e)}"
        SessionManager.set("pdf_processing_error", error_msg)
        SessionManager.add_message("assistant", f"❌ {error_msg}")
        status_container.error(error_msg)


def _update_qa_chain(status_container: DeltaGenerator) -> None:
    """
    문서 인덱싱은 유지한 채 LLM(QA Chain)만 교체합니다.
    """
    selected_model = SessionManager.get("last_selected_model")
    try:
        SessionManager.add_status_log(f"🔄 LLM 교체 중: {selected_model}")
        
        # [Lazy Import]
        from core.model_loader import load_llm
        
        llm = load_llm(selected_model)
        SessionManager.set("llm", llm)
        SessionManager.replace_last_status_log(f"✅ LLM 교체 완료: {selected_model}")

        logger.info(f"LLM updated to: {selected_model}")
        msg = f"✅ QA 시스템이 '{selected_model}' 모델로 업데이트되었습니다."
        SessionManager.add_message("assistant", msg)

    except Exception as e:
        logger.error(f"QA 업데이트 실패: {e}", exc_info=True)
        status_container.error(f"업데이트 실패: {e}")


# --- Callbacks ---
def on_file_upload() -> None:
    """파일 업로드 이벤트 콜백"""
    uploaded_file = st.session_state.get("pdf_uploader")
    if not uploaded_file:
        return

    # [개선] 파일 타입 검사 (MIME 타입 확인)
    if uploaded_file.type != "application/pdf":
        st.error("❌ 올바른 PDF 파일이 아닙니다. PDF 형식의 파일을 업로드해주세요.")
        return

    # [개선] 파일 크기 검사
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"❌ 파일 크기가 너무 큽니다 ({file_size_mb:.2f} MB). {MAX_FILE_SIZE_MB}MB 이하의 파일을 업로드해주세요.")
        return

    # 파일이 변경된 경우에만 처리
    if uploaded_file.name != SessionManager.get("last_uploaded_file_name"):
        # [관리강화] 이전 임시 파일 즉시 삭제
        old_path = SessionManager.get("pdf_file_path")
        if old_path and os.path.exists(old_path):
            try:
                os.remove(old_path)
                logger.info(f"[System] [Cleanup] 이전 파일 삭제: {old_path}")
            except Exception as e:
                logger.warning(f"이전 파일 삭제 실패: {e}")

        SessionManager.set("last_uploaded_file_name", uploaded_file.name)
        
        # [전용 폴더 사용] 안정적인 임시 파일 생성
        try:
            # 절대 경로로 변환
            temp_dir = os.path.abspath(FilePathConstants.TEMP_DIR)
            os.makedirs(temp_dir, exist_ok=True)
            
            # 파일명에 타임스탬프를 넣어 중복 방지 (안전성)
            import time
            safe_name = f"upload_{int(time.time())}.pdf"
            tmp_path = os.path.join(temp_dir, safe_name)
            
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            SessionManager.set("pdf_file_path", tmp_path)
            SessionManager.set("new_file_uploaded", True)
            st.toast(f"문서 업로드 완료: {uploaded_file.name}", icon="📄")
            logger.info(f"[System] [Upload] 파일 저장 완료: {tmp_path}")
        except Exception as e:
            st.error(f"파일 저장 중 오류 발생: {e}")
            logger.error(f"파일 저장 오류: {e}")


def on_model_change() -> None:
    """LLM 모델 변경 이벤트 콜백"""
    selected = st.session_state.get("model_selector")
    last = SessionManager.get("last_selected_model")

    if not selected or "---" in selected or selected == last:
        return

    if not SessionManager.get("is_first_run"):
        SessionManager.add_message("assistant", f"🔄 LLM 변경 요청: {selected}")

    SessionManager.set("last_selected_model", selected)
    # 이미 문서가 처리된 상태라면 QA 체인만 업데이트하면 됨
    if SessionManager.get("pdf_processed"):
        SessionManager.set("needs_qa_chain_update", True)


def on_embedding_change() -> None:
    """임베딩 모델 변경 이벤트 콜백"""
    selected = st.session_state.get("embedding_model_selector")
    last = SessionManager.get("last_selected_embedding_model")

    if not selected or selected == last:
        return

    if not SessionManager.get("is_first_run"):
        SessionManager.add_message("assistant", f"🔄 임베딩 모델 변경 요청: {selected}")

    SessionManager.set("last_selected_embedding_model", selected)
    # 임베딩 모델이 바뀌면 문서를 다시 인덱싱해야 함
    if SessionManager.get("pdf_file_path"):
        SessionManager.set("needs_rag_rebuild", True)


def _render_app_layout(is_skeleton_pass: bool) -> Dict[str, Any]:
    """앱의 전체 레이아웃을 렌더링하고 주요 플레이스홀더를 반환합니다."""
    inject_custom_css()
    
    # 1. 사이드바 렌더링
    if is_skeleton_pass:
        sidebar_placeholders = render_sidebar(
            file_uploader_callback=on_file_upload,
            model_selector_callback=on_model_change,
            embedding_selector_callback=on_embedding_change,
            is_generating=False,
            current_file_name=None,
            current_embedding_model=None
        )
    else:
        sidebar_placeholders = render_sidebar(
            file_uploader_callback=on_file_upload,
            model_selector_callback=on_model_change,
            embedding_selector_callback=on_embedding_change,
            is_generating=st.session_state.get("is_generating_answer", False),
            current_file_name=st.session_state.get("last_uploaded_file_name"),
            current_embedding_model=st.session_state.get("last_selected_embedding_model")
        )
    
    # 2. 메인 영역 레이아웃
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.subheader(StringConstants.MSG_CHAT_TITLE if hasattr(StringConstants, "MSG_CHAT_TITLE") else "💬 채팅")
        render_left_column()
        
    with col_right:
        st.subheader(StringConstants.MSG_PDF_VIEWER_TITLE if hasattr(StringConstants, "MSG_PDF_VIEWER_TITLE") else "📄 PDF 미리보기")
        render_pdf_viewer()
        
    return sidebar_placeholders


def _handle_pending_tasks(status_container: DeltaGenerator) -> None:
    """지연된 무거운 작업(RAG 빌드, 모델 교체 등)을 순차적으로 처리합니다."""
    if SessionManager.get("new_file_uploaded"):
        current_file_path = SessionManager.get("pdf_file_path")
        current_file_name = SessionManager.get("last_uploaded_file_name")
        SessionManager.reset_for_new_file()
        SessionManager.set("pdf_file_path", current_file_path)
        SessionManager.set("last_uploaded_file_name", current_file_name)
        SessionManager.set("new_file_uploaded", False)
        _rebuild_rag_system(status_container)
        st.rerun()

    elif SessionManager.get("needs_rag_rebuild"):
        SessionManager.set("needs_rag_rebuild", False)
        _rebuild_rag_system(status_container)
        st.rerun()

    elif SessionManager.get("needs_qa_chain_update"):
        SessionManager.set("needs_qa_chain_update", False)
        _update_qa_chain(status_container)
        st.rerun()


def main() -> None:
    """메인 애플리케이션 오케스트레이터"""
    # 1. 초기 UI 렌더링 (즉시 실행)
    is_skeleton_pass = "_ui_frame_ready" not in st.session_state
    sidebar_placeholders = _render_app_layout(is_skeleton_pass)

    # 2. UI-First: 뼈대 출력 후 리런하여 데이터 로드 단계 진입
    if is_skeleton_pass:
        st.session_state._ui_frame_ready = True
        st.rerun()

    # 3. 데이터 및 세션 초기화
    SessionManager.init_session()
    status_container = sidebar_placeholders["status_container"]
    SessionManager.set("status_placeholder", status_container)

    # 4. 모델 목록 처리 및 선택기 활성화
    available_models = st.session_state.get("available_models_list")
    if not available_models:
        with sidebar_placeholders["model_selector"]:
            st.selectbox(
                "메인 LLM", ["모델 목록을 불러오는 중..."], index=0, disabled=True, label_visibility="collapsed"
            )
            with st.spinner("Ollama 모델 검색 중..."):
                from core.model_loader import get_available_models
                st.session_state.available_models_list = get_available_models()
        st.rerun()
    else:
        # 정상 모델 선택기 렌더링
        is_ollama_error = available_models[0] == StringConstants.MSG_ERROR_OLLAMA_NOT_RUNNING if hasattr(StringConstants, "MSG_ERROR_OLLAMA_NOT_RUNNING") else False
        actual_models = [] if is_ollama_error else [m for m in available_models if "---" not in m]
        
        last_model = SessionManager.get("last_selected_model")
        if not last_model or (actual_models and last_model not in actual_models):
            last_model = DEFAULT_OLLAMA_MODEL if DEFAULT_OLLAMA_MODEL in actual_models else (actual_models[0] if actual_models else DEFAULT_OLLAMA_MODEL)
            SessionManager.set("last_selected_model", last_model)

        sidebar_placeholders["model_selector"].selectbox(
            "메인 LLM", available_models, 
            index=available_models.index(last_model) if last_model in available_models else 0,
            key="model_selector", on_change=on_model_change, 
            disabled=is_ollama_error, label_visibility="collapsed"
        )

    # 5. 백그라운드 태스크 처리
    _handle_pending_tasks(status_container)

    # 6. 첫 실행 플래그 해제
    if SessionManager.get("is_first_run"):
        SessionManager.set("is_first_run", False)



if __name__ == "__main__":
    main()