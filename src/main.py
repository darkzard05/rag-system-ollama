"""
RAG Chatbot 애플리케이션의 메인 진입점 파일입니다.
"""

import logging
import streamlit as st

from session import SessionManager
from ui import render_sidebar, render_chat_column, render_pdf_viewer
from rag_core import build_rag_pipeline, update_llm_in_pipeline
from model_loader import load_llm, load_embedding_model, is_embedding_model_cached
from config import AVAILABLE_EMBEDDING_MODELS

# --- 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# --- 페이지 설정 ---
st.set_page_config(page_title="RAG Chatbot", layout="wide")


# --- 헬퍼 함수 ---
def _ensure_models_are_loaded(status_container):
    """LLM과 임베딩 모델이 세션에 로드되었는지 확인하고, 없으면 로드합니다."""
    selected_model = SessionManager.get("last_selected_model")
    selected_embedding = SessionManager.get("last_selected_embedding_model")

    if not selected_model:
        st.warning("LLM 모델이 선택되지 않았습니다. 사이드바에서 모델을 선택해주세요.")
        return False
    if not selected_embedding:
        selected_embedding = AVAILABLE_EMBEDDING_MODELS[0]
        SessionManager.set("last_selected_embedding_model", selected_embedding)

    llm = SessionManager.get("llm")
    if not llm or llm.model != selected_model:
        with status_container:
            with st.spinner(f"'{selected_model}' LLM 모델 로딩 중..."):
                llm = load_llm(selected_model)
                SessionManager.set("llm", llm)

    embedder = SessionManager.get("embedder")
    if not embedder or embedder.model_name != selected_embedding:
        spinner_msg = f"'{selected_embedding}' 임베딩 모델 로딩 중..."
        if not is_embedding_model_cached(selected_embedding):
            spinner_msg = f"'{selected_embedding}' 모델을 처음 로드합니다. 시간이 걸릴 수 있습니다..."
        with status_container:
            with st.spinner(spinner_msg):
                embedder = load_embedding_model(selected_embedding)
                SessionManager.set("embedder", embedder)

    return True


def _rebuild_rag_system(status_container):
    """RAG 시스템 재구축을 위한 UI 래퍼 함수."""
    file_name = SessionManager.get("last_uploaded_file_name")
    file_bytes = SessionManager.get("pdf_file_bytes")

    if not all([file_name, file_bytes]):
        status_container.warning("RAG 시스템을 구축하기 위한 파일 정보가 부족합니다.")
        return

    try:
        status_message = f"'{file_name}' 문서 처리 및 RAG 시스템 구축 중..."
        with status_container, st.spinner(status_message):
            if not _ensure_models_are_loaded(status_container):
                return

            llm = SessionManager.get("llm")
            embedder = SessionManager.get("embedder")

            success_message, cache_used = build_rag_pipeline(
                uploaded_file_name=file_name,
                file_bytes=file_bytes,
                llm=llm,
                embedder=embedder,
            )
            if cache_used:
                status_container.info(success_message)

        SessionManager.add_message("assistant", success_message)
        status_container.success("RAG 시스템 구축 완료!")

    except Exception as e:
        error_msg = f"RAG 시스템 구축 중 오류 발생: {e}"
        logging.error(error_msg, exc_info=True)
        SessionManager.set("pdf_processing_error", error_msg)
        SessionManager.add_message("assistant", f"❌ {error_msg}")
        status_container.error(f"오류: {e}")


def _update_qa_chain(status_container):
    """LLM 변경 시 QA 체인 업데이트를 위한 UI 래퍼 함수."""
    selected_model = SessionManager.get("last_selected_model")
    try:
        with status_container, st.spinner(
            f"'{selected_model}' 모델 로드 및 QA 시스템 업데이트 중..."
        ):
            llm = load_llm(selected_model)
            update_llm_in_pipeline(llm)
            success_message = "✅ QA 시스템이 새 모델로 업데이트되었습니다."
            status_container.success(success_message)
            SessionManager.add_message("assistant", success_message)
    except Exception as e:
        error_msg = f"QA 시스템 업데이트 중 오류 발생: {e}"
        logging.error(error_msg, exc_info=True)
        status_container.error(error_msg)
        SessionManager.add_message("assistant", f"❌ {error_msg}")


# --- on_change 콜백 함수 ---
def on_file_upload():
    """파일 업로드 시 실행되는 콜백. 상태를 직접 변경하지 않고 플래그만 설정."""
    uploaded_file = st.session_state.get("pdf_uploader")
    if not uploaded_file:
        return

    # 파일이 변경되었는지 확인
    if uploaded_file.name != SessionManager.get("last_uploaded_file_name"):
        SessionManager.set("last_uploaded_file_name", uploaded_file.name)
        SessionManager.set("pdf_file_bytes", uploaded_file.getvalue())
        SessionManager.set("new_file_uploaded", True)


def on_model_change():
    """LLM 모델 변경 시 실행되는 콜백."""
    selected_model = st.session_state.get("model_selector")
    last_model = SessionManager.get("last_selected_model")

    if "---" in selected_model or not selected_model or selected_model == last_model:
        return

    if not SessionManager.get("is_first_run"):
        SessionManager.add_message(
            "assistant", f"🔄 LLM을 '{selected_model}'(으)로 변경합니다."
        )

    SessionManager.set("last_selected_model", selected_model)
    if SessionManager.get("pdf_processed"):
        SessionManager.set("needs_qa_chain_update", True)


def on_embedding_change():
    """임베딩 모델 변경 시 실행되는 콜백."""
    selected_embedding = st.session_state.get("embedding_model_selector")
    last_embedding = SessionManager.get("last_selected_embedding_model")

    if not selected_embedding or selected_embedding == last_embedding:
        return

    if not SessionManager.get("is_first_run"):
        SessionManager.add_message(
            "assistant", f"🔄 임베딩 모델을 '{selected_embedding}'(으)로 변경합니다."
        )

    SessionManager.set("last_selected_embedding_model", selected_embedding)
    if SessionManager.get("pdf_file_bytes"):
        SessionManager.set("needs_rag_rebuild", True)


# --- 메인 애플리케이션 실행 ---
def main():
    """메인 애플리케이션 실행 함수"""
    SessionManager.init_session()

    status_container = render_sidebar(
        file_uploader_callback=on_file_upload,
        model_selector_callback=on_model_change,
        embedding_selector_callback=on_embedding_change,
    )

    # 새 파일이 업로드되면 관련 상태를 리셋
    if SessionManager.get("new_file_uploaded"):
        SessionManager.reset_for_new_file()
        SessionManager.set("new_file_uploaded", False)

        file_name = SessionManager.get("last_uploaded_file_name")
        SessionManager.add_message("assistant", f"📂 '{file_name}' 파일 업로드 완료.")

    if SessionManager.get("needs_rag_rebuild"):
        SessionManager.set("needs_rag_rebuild", False)
        _rebuild_rag_system(status_container)

    elif SessionManager.get("needs_qa_chain_update"):
        SessionManager.set("needs_qa_chain_update", False)
        _update_qa_chain(status_container)

    col_left, col_right = st.columns([1, 1])
    with col_left:
        render_chat_column()
    with col_right:
        render_pdf_viewer()

    if SessionManager.get("is_first_run"):
        SessionManager.set("is_first_run", False)


if __name__ == "__main__":
    main()
