"""
RAG Chatbot 애플리케이션의 메인 진입점 파일입니다.
"""
import logging
import tempfile
import streamlit as st

# 리팩토링된 모듈 임포트
from session import SessionManager
from ui import render_sidebar, render_chat_column, render_pdf_viewer
from rag_core import (
    process_pdf_and_build_chain, 
    create_qa_chain, 
    load_llm, 
    load_embedding_model, 
    create_vector_store,
    is_embedding_model_cached
)
from config import AVAILABLE_EMBEDDING_MODELS

# --- 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# --- 페이지 설정 ---
st.set_page_config(
    page_title="RAG Chatbot",
    layout="wide"
)

# --- 핸들러 및 헬퍼 함수 ---
def update_qa_system():
    """QA 시스템을 현재 세션 상태에 맞춰 업데이트하는 헬퍼 함수"""
    try:
        llm = SessionManager.get_llm()
        vector_store = SessionManager.get_vector_store()
        doc_splits = SessionManager.get_processed_document_splits()

        if not all([llm, vector_store, doc_splits]):
            st.warning("QA 시스템을 업데이트하기 위한 정보가 부족합니다.")
            return

        qa_chain = create_qa_chain(llm, vector_store, doc_splits)
        SessionManager.set_qa_chain(qa_chain)
        st.rerun()
    except Exception as e:
        st.error(f"QA 시스템 업데이트 중 오류 발생: {e}")
        logging.error("QA 시스템 업데이트 오류", exc_info=True)

def handle_model_change(selected_model: str):
    """모델 변경을 처리하는 핸들러"""
    if "---" in selected_model or \
       not selected_model or \
       selected_model == SessionManager.get_last_selected_model():
        return

    SessionManager.update_model(selected_model)
    
    if SessionManager.get_pdf_processed():
        with st.spinner(f"'{selected_model}' 모델로 QA 시스템 업데이트 중..."):
            llm = load_llm(selected_model)
            SessionManager.set_llm(llm)
            update_qa_system()

def handle_embedding_model_change(selected_embedding_model: str):
    """임베딩 모델 변경을 처리하는 핸들러. 초기 설정 시에는 메시지를 추가하지 않음."""
    last_embedding_model = SessionManager.get_last_selected_embedding_model()

    if not selected_embedding_model or selected_embedding_model == last_embedding_model:
        return

    SessionManager.set_last_selected_embedding_model(selected_embedding_model)
    
    if last_embedding_model is not None:
        SessionManager.add_message("assistant", f"🔄 임베딩 모델을 '{selected_embedding_model}'로 변경했습니다.")

    if SessionManager.get_pdf_processed():
        if not is_embedding_model_cached(selected_embedding_model):
            st.info(f"'{selected_embedding_model}' 모델을 처음 로드합니다. 다운로드가 필요하며 몇 분 정도 소요될 수 있습니다.")

        with st.spinner(f"'{selected_embedding_model}' 임베딩 모델로 QA 시스템 업데이트 중..."):
            embedder = load_embedding_model(selected_embedding_model)
            SessionManager.set_embedder(embedder)
            
            doc_splits = SessionManager.get_processed_document_splits()
            vector_store = create_vector_store(doc_splits, embedder)
            SessionManager.set_vector_store(vector_store)
            
            update_qa_system()


def handle_file_upload(uploaded_file):
    """파일 업로드를 처리하는 핸들러"""
    if uploaded_file.name == SessionManager.get_last_uploaded_file_name():
        return

    file_bytes = uploaded_file.getvalue()
    SessionManager.reset_for_new_file(uploaded_file.name, file_bytes)
    
    try:
        # RAG Core 처리를 위해 임시 파일은 여전히 필요
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_bytes)
            temp_path = tmp_file.name

        SessionManager.add_message("assistant", f"📂 '{uploaded_file.name}' 파일 업로드 완료.")
        
        # PDF 처리
        selected_model = SessionManager.get_last_selected_model()
        selected_embedding_model = SessionManager.get_last_selected_embedding_model() or AVAILABLE_EMBEDDING_MODELS[0]
        
        if not selected_model:
            st.warning("LLM 모델이 선택되지 않았습니다. 사이드바에서 모델을 선택해주세요.")
            return

        # 모델 로드 전 캐시 확인 및 안내 메시지 표시
        if not is_embedding_model_cached(selected_embedding_model):
            st.info(f"'{selected_embedding_model}' 모델을 처음 로드합니다. 다운로드가 필요하며 몇 분 정도 소요될 수 있습니다.")

        with st.spinner(f"'{uploaded_file.name}' 문서 처리 중... 잠시만 기다려주세요."):
            success_message = process_pdf_and_build_chain(
                uploaded_file,
                temp_path, # 임시 파일 경로 전달
                selected_model,
                selected_embedding_model
            )
            SessionManager.add_message("assistant", success_message)
        
        # 임시 파일 삭제
        import os
        os.remove(temp_path)
        
        st.rerun()

    except Exception as e:
        error_msg = f"파일 처리 중 오류 발생: {e}"
        logging.error(error_msg, exc_info=True)
        SessionManager.set_error_state(error_msg)
        st.rerun()

# --- 메인 애플리케이션 실행 ---
def main():
    """메인 애플리케이션 실행 함수"""
    SessionManager.init_session()
    
    render_sidebar(
        uploaded_file_handler=handle_file_upload,
        model_change_handler=handle_model_change,
        embedding_model_change_handler=handle_embedding_model_change
    )

    col_left, col_right = st.columns([1, 1])

    with col_left:
        render_chat_column()

    with col_right:
        render_pdf_viewer()

if __name__ == "__main__":
    main()