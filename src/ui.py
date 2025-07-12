"""
Streamlit UI 컴포넌트 렌더링 함수들을 모아놓은 파일.
"""
import os
import time
import logging
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

from session import SessionManager
from rag_core import get_ollama_models, create_qa_chain
from config import (
    THINK_START_TAG,
    THINK_END_TAG,
    MSG_PREPARING_ANSWER,
    MSG_THINKING,
    MSG_WRITING_ANSWER,
    MSG_NO_THOUGHT_PROCESS,
    MSG_NO_RELATED_INFO,
)

def _process_chat_response(qa_chain, user_input, chat_container):
    """스트리밍 방식으로 LLM 응답을 처리하고 채팅 컨테이너에 표시"""
    with chat_container, st.chat_message("assistant"):
        thought_expander = st.expander("🤔 생각 과정", expanded=False)
        thought_placeholder = thought_expander.empty()
        message_placeholder = st.empty()
        
        message_placeholder.markdown(MSG_PREPARING_ANSWER)
        thought_placeholder.markdown(MSG_NO_THOUGHT_PROCESS)

        try:
            start_time = time.time()
            thought_buffer = ""
            response_buffer = ""
            is_thinking = False
            
            for chunk in qa_chain.stream({"input": user_input}):
                answer_chunk = chunk.get("answer", "")
                if not answer_chunk:
                    continue
                
                if THINK_START_TAG in answer_chunk and not is_thinking:
                    is_thinking = True
                    parts = answer_chunk.split(THINK_START_TAG, 1)
                    response_buffer += parts[0]
                    thought_buffer = parts[1]
                elif THINK_END_TAG in answer_chunk:
                    is_thinking = False
                    parts = answer_chunk.split(THINK_END_TAG, 1)
                    thought_buffer += parts[0]
                    response_buffer += parts[1]
                elif is_thinking:
                    thought_buffer += answer_chunk
                else:
                    response_buffer += answer_chunk
                
                if thought_buffer.strip():
                    thought_placeholder.markdown(thought_buffer + "▌")
                if not is_thinking and thought_buffer.strip():
                     thought_placeholder.markdown(thought_buffer)

                message_placeholder.markdown(response_buffer + "▌")

            # 최종 내용 업데이트
            if thought_buffer.strip():
                thought_placeholder.markdown(thought_buffer)
            
            final_answer = response_buffer.strip()
            if not final_answer:
                final_answer = MSG_NO_RELATED_INFO
            
            message_placeholder.markdown(final_answer)
            SessionManager.add_message("assistant", final_answer)
            logging.info(f"LLM 답변 생성 완료 (소요 시간: {time.time() - start_time:.2f}초)")

        except Exception as e:
            error_msg = f"답변 생성 중 오류 발생: {str(e)}"
            logging.error(error_msg, exc_info=True)
            message_placeholder.error(error_msg)
            SessionManager.add_message("assistant", f"❌ {error_msg}")

def render_sidebar(uploaded_file_handler, model_change_handler):
    """사이드바 UI를 렌더링하고 사용자 입력을 처리합니다."""
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 모델 선택
        try:
            models = get_ollama_models()
            if not models:
                st.warning("Ollama 모델을 찾을 수 없습니다. Ollama 서버가 실행 중인지 확인해주세요.")
                return

            last_model = st.session_state.get("last_selected_model")
            current_model_index = models.index(last_model) if last_model and last_model in models else 0
            
            selected_model = st.selectbox(
                "Ollama 모델 선택",
                models,
                index=current_model_index,
                key="model_selector"
            )
            model_change_handler(selected_model)

        except Exception as e:
            st.error(f"Ollama 모델 로드 실패: {e}")

        # 파일 업로더
        uploaded_file = st.file_uploader("PDF 파일 업로드", type="pdf")
        if uploaded_file:
            uploaded_file_handler(uploaded_file)
        
        st.divider()
        
        # PDF 뷰어 설정
        st.session_state.resolution_boost = st.slider("해상도", 1, 10, 1)
        st.session_state.pdf_width = st.slider("PDF 너비", 100, 1000, 1000)
        st.session_state.pdf_height = st.slider("PDF 높이", 100, 10000, 1000)

def render_pdf_viewer():
    """PDF 뷰어 컬럼을 렌더링합니다."""
    st.subheader("📄 PDF 미리보기")
    
    temp_pdf_path = st.session_state.get("temp_pdf_path")
    if temp_pdf_path and os.path.exists(temp_pdf_path):
        try:
            pdf_viewer(
                input=temp_pdf_path,
                width=st.session_state.pdf_width,
                height=st.session_state.pdf_height,
                key=f"pdf_viewer_{st.session_state.last_uploaded_file_name}",
                resolution_boost=st.session_state.resolution_boost
            )
        except Exception as e:
            st.error(f"PDF 미리보기 중 오류 발생: {str(e)}")
            logging.error("PDF 미리보기 오류", exc_info=True)

def render_chat_column():
    """채팅 컬럼을 렌더링하고 채팅 로직을 처리합니다."""
    st.subheader("💬 채팅")
    
    chat_container = st.container(height=650, border=True)
    
    # 기존 메시지 표시
    for message in st.session_state.get("messages", []):
        with chat_container, st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)
    
    # 사용자 입력 처리
    if user_input := st.chat_input(
        "PDF 내용에 대해 질문해보세요.",
        disabled=not SessionManager.is_ready_for_chat()
    ):
        SessionManager.add_message("user", user_input)
        with chat_container, st.chat_message("user"):
            st.markdown(user_input)
        
        qa_chain = st.session_state.get("qa_chain")
        if qa_chain:
            _process_chat_response(qa_chain, user_input, chat_container)
        else:
            st.error("QA 시스템이 준비되지 않았습니다. PDF를 먼저 처리해주세요.")
