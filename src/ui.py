"""
Streamlit UI 컴포넌트 렌더링 함수들을 모아놓은 파일.
"""
from email import message
import os
import time
import logging
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

from session import SessionManager
from rag_core import get_available_models
from config import (
    AVAILABLE_EMBEDDING_MODELS,
    OLLAMA_MODEL_NAME,
    THINK_START_TAG,
    THINK_END_TAG,
    MSG_PREPARING_ANSWER,
    MSG_THINKING,
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
        thought_placeholder.markdown(MSG_THINKING)

        try:
            thought_buffer = ""
            response_buffer = ""
            is_thinking = False
            
            start_time = time.time()
            
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

                # 답변 placeholder의 내용을 결정
                current_message = response_buffer
                if not current_message.strip() and is_thinking:
                    current_message = MSG_THINKING
                
                message_placeholder.markdown(current_message + "▌")

            end_time = time.time()
            elapsed_time = end_time - start_time

            # 최종 내용 업데이트
            if thought_buffer.strip():
                thought_placeholder.markdown(thought_buffer)
            else:
                # 스트림이 모두 끝난 후에도 생각 내용이 없으면 그때 메시지 표시
                thought_placeholder.markdown(MSG_NO_THOUGHT_PROCESS)
            
            final_answer = response_buffer.strip()
            if not final_answer:
                final_answer = MSG_NO_RELATED_INFO
            
            message_placeholder.markdown(final_answer)
            SessionManager.add_message("assistant", final_answer)
            
            # 답변 생성 시간 및 글자 수 로깅
            logging.info(
                f"LLM 답변 생성 완료. "
                f"소요 시간: {elapsed_time:.2f}초, "
                f"답변 길이: {len(final_answer)}자"
            )

        except Exception as e:
            error_msg = f"답변 생성 중 오류 발생: {str(e)}"
            logging.error(error_msg, exc_info=True)
            message_placeholder.error(error_msg)
            SessionManager.add_message("assistant", f"❌ {error_msg}")

def render_sidebar(uploaded_file_handler, model_change_handler, embedding_model_change_handler):
    """사이드바 UI를 렌더링하고 사용자 입력을 처리합니다."""
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # --- 파일 업로더 ---
        uploaded_file = st.file_uploader("PDF 파일 업로드", type="pdf")
        if uploaded_file:
            uploaded_file_handler(uploaded_file)
        
        st.divider()

        # --- LLM 모델 선택 (동적 목록) ---
        available_models = get_available_models()
        
        last_model = SessionManager.get_last_selected_model()
        # 마지막으로 선택한 모델이 현재 사용 가능한 목록에 없으면, 목록의 첫 번째 모델을 기본값으로 사용
        if last_model not in available_models:
            last_model = available_models[0] if available_models else OLLAMA_MODEL_NAME

        current_model_index = available_models.index(last_model) if last_model in available_models else 0
        
        selected_model = st.selectbox(
            "LLM 모델 선택",
            available_models,
            index=current_model_index,
            key="model_selector"
        )
        model_change_handler(selected_model)

        # --- 임베딩 모델 선택 ---
        last_embedding_model = SessionManager.get_last_selected_embedding_model() or AVAILABLE_EMBEDDING_MODELS[0]
        current_embedding_model_index = AVAILABLE_EMBEDDING_MODELS.index(last_embedding_model) if last_embedding_model in AVAILABLE_EMBEDDING_MODELS else 0
        
        selected_embedding_model = st.selectbox(
            "임베딩 모델 선택",
            AVAILABLE_EMBEDDING_MODELS,
            index=current_embedding_model_index,
            key="embedding_model_selector"
        )
        embedding_model_change_handler(selected_embedding_model)

        st.divider()

        # --- PDF 뷰어 설정 ---
        resolution_boost = st.slider("해상도", 1, 10, SessionManager.get_resolution_boost())
        SessionManager.set_resolution_boost(resolution_boost)
        pdf_width = st.slider("PDF 너비", 100, 1000, SessionManager.get_pdf_width())
        SessionManager.set_pdf_width(pdf_width)
        pdf_height = st.slider("PDF 높이", 100, 10000, SessionManager.get_pdf_height())
        SessionManager.set_pdf_height(pdf_height)

def render_pdf_viewer():
    """PDF 뷰어 컬럼을 렌더링합니다."""
    st.subheader("📄 PDF 미리보기")
    
    temp_pdf_path = SessionManager.get_temp_pdf_path()
    if temp_pdf_path and os.path.exists(temp_pdf_path):
        try:
            pdf_viewer(
                input=temp_pdf_path,
                width=SessionManager.get_pdf_width(),
                height=SessionManager.get_pdf_height(),
                key=f"pdf_viewer_{SessionManager.get_last_uploaded_file_name()}",
                resolution_boost=SessionManager.get_resolution_boost()
            )
        except Exception as e:
            st.error(f"PDF 미리보기 중 오류 발생: {str(e)}")
            logging.error("PDF 미리보기 오류", exc_info=True)

def render_chat_column():
    """채팅 컬럼을 렌더링하고 채팅 로직을 처리합니다."""
    st.subheader("💬 채팅")
    
    chat_container = st.container(height=650, border=True)
    
    # 기존 메시지 표시
    for message in SessionManager.get_messages():
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
        
        qa_chain = SessionManager.get_qa_chain()
        if qa_chain:
            _process_chat_response(qa_chain, user_input, chat_container)
        else:
            st.error("QA 시스템이 준비되지 않았습니다. PDF를 먼저 처리해주세요.")

    # 초기 안내 메시지
    if not SessionManager.get_messages():
        with chat_container:
            st.info(
                "**RAG-Chat에 오신 것을 환영합니다!**\n\n"
                "사이드바에서 PDF 파일을 업로드하여 문서 내용에 대한 대화를 시작해보세요."
            )
            
            st.markdown(
                """
                **💡 사용 가이드:**
                - **PDF 업로드:** 좌측 사이드바에서 분석하고 싶은 PDF를 업로드하세요.
                - **모델 선택:** 로컬 `Ollama` 모델 또는 `Gemini` API 모델을 선택할 수 있습니다.
                - **질문하기:** 문서 처리가 완료되면, 내용에 대해 자유롭게 질문할 수 있습니다.
                - **PDF 뷰어:** 우측에서 원본 문서를 함께 보며 대화할 수 있습니다. (해상도/크기 조절 가능)
                """
            )

            with st.expander("⚠️ 알아두실 점"):
                st.warning(
                    "**답변의 정확성:** 답변은 업로드된 PDF 내용만을 기반으로 생성되며, 사실이 아닐 수 있습니다."
                )
                st.warning(
                    "**개인정보:** Gemini 모델 사용 시, 질문 내용이 Google 서버로 전송될 수 있습니다."
                )
                st.warning(
                    "**초기 로딩:** 임베딩 모델을 처음 사용하면 다운로드에 몇 분이 소요될 수 있습니다."
                )