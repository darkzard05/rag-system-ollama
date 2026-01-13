"""
RAG Chatbot 애플리케이션의 메인 진입점 파일입니다.
Streamlit 프레임워크를 기반으로 UI를 구성하고 세션 상태를 관리합니다.
"""

import logging
from typing import Any

import nest_asyncio
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from config import AVAILABLE_EMBEDDING_MODELS
from model_loader import load_embedding_model, load_llm, is_embedding_model_cached
from rag_core import build_rag_pipeline
from session import SessionManager
from ui import render_left_column, render_pdf_viewer, render_sidebar

# 상수 정의
PAGE_TITLE = "RAG Chatbot"
LAYOUT = "wide"
MAX_FILE_SIZE_MB = 50  # 최대 파일 크기 제한 (MB)

# 비동기 패치 적용 (최상단 실행)
nest_asyncio.apply()

# 로깅 설정
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
logger = logging.getLogger(__name__)

# Streamlit 페이지 설정
st.set_page_config(page_title=PAGE_TITLE, layout=LAYOUT)


def _ensure_models_are_loaded(status_container: DeltaGenerator) -> bool:
    """
    선택된 LLM 및 임베딩 모델이 세션에 로드되어 있는지 확인하고, 필요 시 로드합니다.
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
        # LLM 로드 상태 확인 및 로드
        current_llm = SessionManager.get("llm")
        if not current_llm or getattr(current_llm, "model", None) != selected_model:
            with status_container:
                with st.spinner(f"🧠 LLM 모델 로딩 중: '{selected_model}'..."):
                    new_llm = load_llm(selected_model)
                    SessionManager.set("llm", new_llm)

        # 임베딩 모델 로드 상태 확인 및 로드
        current_embedder = SessionManager.get("embedder")
        if not current_embedder or getattr(current_embedder, "model_name", None) != selected_embedding:
            msg = f"🧮 임베딩 모델 로딩 중: '{selected_embedding}'..."
            if not is_embedding_model_cached(selected_embedding):
                msg += " (최초 다운로드)"

            with status_container:
                with st.spinner(msg):
                    new_embedder = load_embedding_model(selected_embedding)
                    SessionManager.set("embedder", new_embedder)

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
    file_bytes = SessionManager.get("pdf_file_bytes")

    if not file_name or not file_bytes:
        return

    # [중복 실행 방지] 이미 해당 파일에 대한 처리가 완료되었는지 확인
    # - pdf_processed가 True이고
    # - 에러가 없으며
    # - 벡터 스토어 객체가 메모리에 존재하는 경우
    # 재구축을 건너뜁니다.
    if (SessionManager.get("pdf_processed") 
        and not SessionManager.get("pdf_processing_error") 
        and SessionManager.get("vector_store") is not None):
        logger.debug(f"파일 '{file_name}'에 대한 RAG 파이프라인이 이미 구축되어 있습니다. 재구축을 건너뜁니다.")
        return

    try:
        if not _ensure_models_are_loaded(status_container):
            return

        embedder = SessionManager.get("embedder")

        with status_container:
            with st.spinner(f"⚙️ 문서 분석 및 인덱싱 중: '{file_name}'..."):
                # RAG 파이프라인 빌드 (시간이 소요될 수 있음)
                success_message, cache_used = build_rag_pipeline(
                    uploaded_file_name=file_name,
                    file_bytes=file_bytes,
                    embedder=embedder,
                )

                if cache_used:
                    status_container.info("✅ 캐시된 인덱스를 로드했습니다.")
                else:
                    status_container.success("✅ 새로운 문서 인덱싱 완료.")

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
        with status_container:
            with st.spinner(f"🔄 LLM 교체 중: '{selected_model}'..."):
                llm = load_llm(selected_model)
                SessionManager.set("llm", llm)

        logger.info(f"LLM updated to: {selected_model}")
        msg = f"✅ QA 시스템이 '{selected_model}' 모델로 업데이트되었습니다."
        status_container.success(msg)
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

    # [개선] 파일 크기 검사
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"❌ 파일 크기가 너무 큽니다 ({file_size_mb:.2f} MB). {MAX_FILE_SIZE_MB}MB 이하의 파일을 업로드해주세요.")
        return

    # 파일이 변경된 경우에만 처리
    if uploaded_file.name != SessionManager.get("last_uploaded_file_name"):
        SessionManager.set("last_uploaded_file_name", uploaded_file.name)
        # 주의: 큰 파일의 경우 getvalue()는 메모리를 많이 소모할 수 있음
        SessionManager.set("pdf_file_bytes", uploaded_file.getvalue())
        SessionManager.set("new_file_uploaded", True)


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
    if SessionManager.get("pdf_file_bytes"):
        SessionManager.set("needs_rag_rebuild", True)


def main() -> None:
    """메인 애플리케이션 로직"""
    SessionManager.init_session()

    # 사이드바 렌더링 및 상태 컨테이너 확보
    status_container = render_sidebar(
        file_uploader_callback=on_file_upload,
        model_selector_callback=on_model_change,
        embedding_selector_callback=on_embedding_change,
    )

    # 상태 플래그에 따른 작업 수행 (우선순위: 새 파일 > 임베딩 변경 > 모델 변경)
    if SessionManager.get("new_file_uploaded"):
        SessionManager.reset_for_new_file()
        SessionManager.set("new_file_uploaded", False)
        _rebuild_rag_system(status_container)

    elif SessionManager.get("needs_rag_rebuild"):
        SessionManager.set("needs_rag_rebuild", False)
        _rebuild_rag_system(status_container)

    elif SessionManager.get("needs_qa_chain_update"):
        SessionManager.set("needs_qa_chain_update", False)
        _update_qa_chain(status_container)

    # 메인 UI 레이아웃 (채팅창 + PDF 뷰어)
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