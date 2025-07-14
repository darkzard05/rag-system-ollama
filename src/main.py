"""
RAG Chatbot 애플리케이션의 메인 진입점 파일입니다.
"""
import os
import logging
import tempfile
import streamlit as st

# 리팩토링된 모듈 임포트
from session import SessionManager
from ui import render_sidebar, render_chat_column, render_pdf_viewer
from rag_core import process_pdf_and_build_chain, create_qa_chain, load_llm

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

# --- 핸들러 함수 ---
def handle_gemini_api_key_change(api_key: str):
    """Gemini API 키 변경을 처리하는 핸들러"""
    st.session_state.gemini_api_key = api_key

def handle_model_change(selected_model: str):
    """모델 변경을 처리하는 핸들러"""
    if not selected_model or selected_model == st.session_state.get("last_selected_model"):
        return

    SessionManager.update_model(selected_model)
    
    if st.session_state.get("pdf_processed"):
        with st.spinner(f"'{selected_model}' 모델로 QA 시스템 업데이트 중..."):
            try:
                gemini_api_key = st.session_state.get("gemini_api_key")
                llm = load_llm(selected_model, gemini_api_key)
                st.session_state.llm = llm
                qa_chain = create_qa_chain(
                    llm,
                    st.session_state.vector_store,
                    st.session_state.processed_document_splits
                )
                st.session_state.qa_chain = qa_chain
                st.rerun()
            except Exception as e:
                st.error(f"모델 변경 중 오류 발생: {e}")
                logging.error("모델 변경 핸들러 오류", exc_info=True)

def handle_file_upload(uploaded_file):
    """파일 업로드를 처리하는 핸들러"""
    if uploaded_file.name == st.session_state.get("last_uploaded_file_name"):
        return

    # 이전 임시 파일 정리
    if st.session_state.get("temp_pdf_path") and os.path.exists(st.session_state.temp_pdf_path):
        try:
            os.remove(st.session_state.temp_pdf_path)
        except Exception as e:
            logging.warning(f"이전 임시 파일 삭제 실패: {e}")

    SessionManager.reset_for_new_file(uploaded_file)

    try:
        # 새 임시 파일 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            st.session_state.temp_pdf_path = tmp_file.name
        
        SessionManager.add_message("assistant", f"📂 '{uploaded_file.name}' 파일 업로드 완료.")
        
        # PDF 처리
        selected_model = st.session_state.get("last_selected_model")
        if not selected_model:
            st.warning("모델이 선택되지 않았습니다. 사이드바에서 모델을 선택해주세요.")
            return
        
        gemini_api_key = st.session_state.get("gemini_api_key")

        with st.spinner(f"'{uploaded_file.name}' 문서 처리 중... 잠시만 기다려주세요."):
            success_message = process_pdf_and_build_chain(
                uploaded_file,
                st.session_state.temp_pdf_path,
                selected_model,
                gemini_api_key
            )
        SessionManager.add_message("assistant", success_message)
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
        gemini_api_key_handler=handle_gemini_api_key_change
    )

    col_left, col_right = st.columns([1, 1])

    with col_left:
        render_chat_column()

    with col_right:
        render_pdf_viewer()

if __name__ == "__main__":
    main()
