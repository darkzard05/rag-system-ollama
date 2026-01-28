"""
RAG Chatbot 애플리케이션의 메인 진입점 파일입니다.
Streamlit 프레임워크를 기반으로 UI를 구성하고 세션 상태를 관리합니다.
"""

import logging
import tempfile
import os
from typing import Any
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

from common.config import AVAILABLE_EMBEDDING_MODELS
from common.constants import StringConstants, FilePathConstants
from core.model_loader import load_embedding_model, load_llm, is_embedding_model_cached
from core.rag_core import build_rag_pipeline
from core.session import SessionManager
from ui.ui import render_left_column, render_pdf_viewer, render_sidebar, _render_status_box, inject_custom_css
from services.optimization.memory_optimizer import get_memory_optimizer

# 상수 정의
PAGE_TITLE = StringConstants.PAGE_TITLE
LAYOUT = StringConstants.LAYOUT
MAX_FILE_SIZE_MB = StringConstants.MAX_FILE_SIZE_MB

# 비동기 패치 적용 (최상단 실행)
nest_asyncio.apply()

@st.cache_resource
def get_and_start_memory_optimizer():
    """메모리 최적화 서비스를 단 한 번만 초기화하고 시작합니다."""
    optimizer = get_memory_optimizer()
    optimizer.start()
    return optimizer

# 메모리 최적화 서비스 시작 (캐싱 적용)
memory_optimizer = get_and_start_memory_optimizer()

# Streamlit 페이지 설정
st.set_page_config(page_title=PAGE_TITLE, layout=LAYOUT)


import threading

def _ensure_models_are_loaded(status_container: DeltaGenerator) -> bool:
    """
    선택된 LLM 및 임베딩 모델을 순차적으로 로드하여 안정성을 확보합니다.
    (병렬 로딩은 GPU 자원 경합으로 인해 TTFT를 증가시킬 수 있어 순차 로딩 권장)
    """
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
        # [메모리 최적화] 이전 임시 파일이 있다면 삭제하여 디스크 공간 확보
        old_path = SessionManager.get("pdf_file_path")
        if old_path and os.path.exists(old_path):
            try:
                os.remove(old_path)
                logger.info(f"이전 임시 파일 삭제 완료: {old_path}")
            except Exception as e:
                logger.warning(f"이전 임시 파일 삭제 실패: {e}")

        SessionManager.set("last_uploaded_file_name", uploaded_file.name)
        
        # [메모리 최적화] 파일을 임시 경로에 저장하고 경로만 세션에 유지
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_path = tmp_file.name
            
            SessionManager.set("pdf_file_path", tmp_path)
            SessionManager.set("new_file_uploaded", True)
            st.toast(f"문서 업로드 완료: {uploaded_file.name}", icon="📄")
            logger.info(f"새 임시 파일 저장 완료: {tmp_path}")
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


def main() -> None:
    """메인 애플리케이션 로직 (최적화된 선형 구조)"""
    SessionManager.init_session()
    
    # [스타일링] 전역 CSS 주입 (레이아웃 틀어짐 방지)
    inject_custom_css()

    # 1. 사이드바 및 상태 컨테이너 렌더링
    status_container = render_sidebar(
        file_uploader_callback=on_file_upload,
        model_selector_callback=on_model_change,
        embedding_selector_callback=on_embedding_change,
    )
    
    # 2. 상태 변경 작업 수행 (메인 UI 렌더링 전 모든 로직 처리)
    # 이 과정에서 발생하는 데이터 변경은 아래 3번 단계에서 즉시 반영됨
    has_changed = False
    
    if SessionManager.get("new_file_uploaded"):
        current_file_path = SessionManager.get("pdf_file_path")
        current_file_name = SessionManager.get("last_uploaded_file_name")
        
        SessionManager.reset_for_new_file()
        SessionManager.set("pdf_file_path", current_file_path)
        SessionManager.set("last_uploaded_file_name", current_file_name)
        SessionManager.set("new_file_uploaded", False)
        
        _rebuild_rag_system(status_container)
        has_changed = True

    elif SessionManager.get("needs_rag_rebuild"):
        SessionManager.set("needs_rag_rebuild", False)
        _rebuild_rag_system(status_container)
        has_changed = True

    elif SessionManager.get("needs_qa_chain_update"):
        SessionManager.set("needs_qa_chain_update", False)
        _update_qa_chain(status_container)
        has_changed = True

    # 3. 메인 UI 레이아웃 (채팅창 + PDF 뷰어)
    # 위에서 추가된 메시지나 상태가 이 단계에서 자연스럽게 포함되어 렌더링됨
    col_left, col_right = st.columns([1, 1])

    with col_left:
        render_left_column()

    with col_right:
        render_pdf_viewer()

    # 첫 실행 플래그 해제
    if SessionManager.get("is_first_run"):
        SessionManager.set("is_first_run", False)



if __name__ == "__main__":
    main()