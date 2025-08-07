"""
RAG Chatbot 애플리케이션의 메인 진입점 파일입니다.
"""
import logging
import tempfile
import time
import streamlit as st
import os

from session import SessionManager
from ui import render_sidebar, render_chat_column, render_pdf_viewer
from rag_core import (
    process_pdf_and_build_chain, 
    create_qa_chain, 
    create_vector_store,
    is_embedding_model_cached,
    load_llm,
    load_embedding_model
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
def _ensure_models_are_loaded(status_container):
    """LLM과 임베딩 모델이 세션에 로드되었는지 확인하고, 없으면 로드합니다."""
    selected_model = SessionManager.get("last_selected_model")
    selected_embedding_model = SessionManager.get("last_selected_embedding_model") or AVAILABLE_EMBEDDING_MODELS[0]

    if not SessionManager.get("llm"):
        status_container.update(label=f"'{selected_model}' LLM 모델 로딩 중...")
        llm = load_llm(selected_model)
        SessionManager.set("llm", llm)
        status_container.update(label=f"'{selected_model}' LLM 모델 로드 완료.")

    if not SessionManager.get("embedder"):
        if not is_embedding_model_cached(selected_embedding_model):
            status_container.update(label=f"'{selected_embedding_model}' 모델을 처음 로드합니다. 다운로드가 필요하며 몇 분 정도 소요될 수 있습니다.")
        else:
            status_container.update(label=f"'{selected_embedding_model}' 임베딩 모델 로딩 중...")
        embedder = load_embedding_model(selected_embedding_model)
        SessionManager.set("embedder", embedder)
        status_container.update(label=f"'{selected_embedding_model}' 임베딩 모델 로드 완료.")

def update_qa_system():
    """QA 시스템을 현재 세션 상태에 맞춰 업데이트하는 헬퍼 함수"""
    try:
        llm = SessionManager.get("llm")
        vector_store = SessionManager.get("vector_store")
        doc_splits = SessionManager.get("processed_document_splits")

        if not all([llm, vector_store, doc_splits]):
            st.warning("QA 시스템을 업데이트하기 위한 정보가 부족합니다.")
            return

        with st.status("QA 시스템 업데이트 중...", expanded=False) as status:
            status.update(label="새로운 LLM으로 QA 체인을 재구성하고 있습니다...")
            qa_chain = create_qa_chain(llm, vector_store, doc_splits)
            SessionManager.set("qa_chain", qa_chain)
            status.update(label="QA 시스템 업데이트 완료!", state="complete", expanded=False)
        
        # 잠시 딜레이를 주어 사용자가 완료 메시지를 인지할 시간을 줍니다.
        time.sleep(1)
        st.rerun()

    except Exception as e:
        st.error(f"QA 시스템 업데이트 중 오류 발생: {e}")
        logging.error("QA 시스템 업데이트 오류", exc_info=True)

def handle_model_change(selected_model: str):
    """모델 변경을 처리하는 핸들러"""
    last_model = SessionManager.get("last_selected_model")
    if "---" in selected_model or not selected_model or selected_model == last_model:
        return

    if last_model is not None:
        SessionManager.add_message("assistant", f"🔄 LLM을 '{selected_model}'(으)로 변경합니다.")
    SessionManager.set("last_selected_model", selected_model)
    
    if SessionManager.get("pdf_processed"):
        with st.status(f"'{selected_model}' 모델 로드 및 시스템 업데이트 중...", expanded=True) as status:
            status.update(label=f"'{selected_model}' 모델을 로드하는 중...")
            llm = load_llm(selected_model)
            SessionManager.set("llm", llm)
            status.update(label="모델 로드 완료.")
            
            # update_qa_system() 호출 대신 직접 로직 수행
            status.update(label="새로운 LLM으로 QA 체인을 재구성하고 있습니다...")
            vector_store = SessionManager.get("vector_store")
            doc_splits = SessionManager.get("processed_document_splits")
            qa_chain = create_qa_chain(llm, vector_store, doc_splits)
            SessionManager.set("qa_chain", qa_chain)
            status.update(label="QA 시스템 업데이트 완료!", state="complete", expanded=False)

        time.sleep(1)
        st.rerun()

def handle_embedding_model_change(selected_embedding_model: str):
    """임베딩 모델 변경을 처리하는 핸들러."""
    last_embedding_model = SessionManager.get("last_selected_embedding_model")
    if not selected_embedding_model or selected_embedding_model == last_embedding_model:
        return

    if last_embedding_model is not None:
        SessionManager.add_message("assistant", f"🔄 임베딩 모델을 '{selected_embedding_model}'(으)로 변경합니다.")
    
    SessionManager.set("last_selected_embedding_model", selected_embedding_model)

    if SessionManager.get("pdf_processed"):
        st.info("임베딩 모델이 변경되어 문서를 다시 처리합니다. 잠시만 기다려주세요...")
        
        file_name = SessionManager.get("last_uploaded_file_name")
        file_bytes = SessionManager.get("pdf_file_bytes")
        llm = SessionManager.get("llm")

        if not all([file_name, file_bytes, llm]):
            st.warning("문서를 다시 처리하기 위한 정보가 부족합니다.")
            return

        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file_bytes)
                temp_path = tmp_file.name

            with st.status(f"새 임베딩 모델로 RAG 시스템 재구축 중...", expanded=True) as status:
                
                def progress_callback(message):
                    status.update(label=message)

                if not is_embedding_model_cached(selected_embedding_model):
                    status.update(label=f"'{selected_embedding_model}' 모델을 처음 로드합니다. 다운로드가 필요하며 몇 분 정도 소요될 수 있습니다.")
                
                embedder = load_embedding_model(selected_embedding_model)
                SessionManager.set("embedder", embedder)
                status.update(label=f"'{selected_embedding_model}' 임베딩 모델 로드 완료.")

                success_message, cache_used = process_pdf_and_build_chain(
                    uploaded_file_name=file_name,
                    file_bytes=file_bytes,
                    temp_pdf_path=temp_path,
                    llm=llm,
                    embedder=embedder,
                    progress_callback=progress_callback
                )
                if cache_used:
                    st.info(success_message)
                
                SessionManager.add_message("assistant", "✅ 새 임베딩 모델로 시스템이 업데이트되었습니다.")
                status.update(label="시스템 재구축 완료!", state="complete", expanded=False)

            time.sleep(1)
            st.rerun()

        except Exception as e:
            error_msg = f"임베딩 모델 변경 후 재처리 중 오류 발생: {e}"
            logging.error(error_msg, exc_info=True)
            SessionManager.add_message("assistant", f"❌ {error_msg}")
            st.rerun()
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
                logging.info(f"임시 파일 '{temp_path}' 삭제 완료.")

def handle_file_upload(uploaded_file):
    """파일 업로드를 처리하는 핸들러"""
    if uploaded_file.name == SessionManager.get("last_uploaded_file_name"):
        return

    # 새 파일 업로드 시 세션 상태 리셋 (모델 관련 상태는 보존)
    preserved_model = SessionManager.get("last_selected_model")
    preserved_embedding_model = SessionManager.get("last_selected_embedding_model")
    preserved_llm = SessionManager.get("llm")
    preserved_embedder = SessionManager.get("embedder")
    
    SessionManager.reset_all_state()
    
    SessionManager.set("last_selected_model", preserved_model)
    SessionManager.set("last_selected_embedding_model", preserved_embedding_model)
    SessionManager.set("llm", preserved_llm)
    SessionManager.set("embedder", preserved_embedder)
    
    # 새 파일 정보 설정
    file_bytes = uploaded_file.getvalue()
    SessionManager.set("last_uploaded_file_name", uploaded_file.name)
    SessionManager.set("pdf_file_bytes", file_bytes)
    
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_bytes)
            temp_path = tmp_file.name

        SessionManager.add_message("assistant", f"📂 '{uploaded_file.name}' 파일 업로드 완료.")
        
        if not SessionManager.get("last_selected_model"):
            st.warning("LLM 모델이 선택되지 않았습니다. 사이드바에서 모델을 선택해주세요.")
            return

        with st.status(f"'{uploaded_file.name}' 문서 처리 및 RAG 시스템 구축 중...", expanded=True) as status:
            
            def progress_callback(message):
                status.update(label=message)

            _ensure_models_are_loaded(status)

            success_message, cache_used = process_pdf_and_build_chain(
                uploaded_file_name=uploaded_file.name,
                file_bytes=file_bytes,
                temp_pdf_path=temp_path,
                llm=SessionManager.get("llm"),
                embedder=SessionManager.get("embedder"),
                progress_callback=progress_callback
            )
            if cache_used:
                st.info(success_message)
            
            SessionManager.add_message("assistant", success_message)
            status.update(label="RAG 시스템 구축 완료!", state="complete", expanded=False)
        
        time.sleep(1)
        st.rerun()

    except Exception as e:
        error_msg = f"파일 처리 중 오류 발생: {e}"
        logging.error(error_msg, exc_info=True)
        SessionManager.set("pdf_processing_error", error_msg)
        SessionManager.add_message("assistant", f"❌ {error_msg}")
        st.rerun()
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            logging.info(f"임시 파일 '{temp_path}' 삭제 완료.")

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