import torch
torch.classes.__path__ = [] # 호환성 문제 해결을 위한 임시 조치
import tempfile
import os
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import logging
from utils import (
    init_session_state,
    reset_session_state,
    get_ollama_models,
    process_pdf,
)

# 페이지 설정
st.set_page_config(page_title="RAG Chatbot", layout="wide")

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# 세션 상태 초기화
init_session_state()

# 사이드바 설정
with st.sidebar:
    st.header("📄 RAG Chatbot with Ollama LLM")
    try:
        models = get_ollama_models()
        current_model_index = models.index(st.session_state.last_selected_model) if st.session_state.last_selected_model in models else 0
        selected_model = st.selectbox(
            "Select an Ollama model",
            models,
            index=current_model_index
        ) if models else st.text("Failed to load Ollama models.")
    except Exception as e:
        st.error(f"Failed to load Ollama models: {e}")
        st.warning("Ollama가 설치되어 있는지, Ollama 서버가 실행 중인지 확인해주세요.")
        selected_model = None

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    # PDF 뷰어 설정
    st.divider()
    resolution_boost = st.slider(label="Resolution boost", min_value=1, max_value=10, value=1)
    width = st.slider(label="PDF width", min_value=100, max_value=1000, value=1000)
    height = st.slider(label="PDF height", min_value=-1, max_value=10000, value=1000)

# 레이아웃 설정
col_left, col_right = st.columns([1, 1])

# 오른쪽 컬럼: PDF 미리보기
with col_right:
    # PDF 미리보기
    if uploaded_file and uploaded_file.name != st.session_state.get("last_uploaded_file_name"):
        if st.session_state.temp_pdf_path and os.path.exists(st.session_state.temp_pdf_path):
            try:
                os.remove(st.session_state.temp_pdf_path)
                logging.info("이전 임시 PDF 파일 삭제 성공")
            except Exception as e:
                logging.warning(f"이전 임시 PDF 파일 삭제 실패: {e}")
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                st.session_state.temp_pdf_path = tmp.name
                logging.info(f"임시 PDF 파일 생성 성공: {st.session_state.temp_pdf_path}")
        except Exception as e:
            st.error(f"임시 PDF 파일 생성 실패: {e}")
            st.session_state.temp_pdf_path = None

    # PDF 미리보기 표시
    if st.session_state.temp_pdf_path and os.path.exists(st.session_state.temp_pdf_path):
        try:
            pdf_viewer(
                input=st.session_state.temp_pdf_path,
                width=width,
                height=height,
                key=f'pdf_viewer_{os.path.basename(st.session_state.temp_pdf_path)}',
                resolution_boost=resolution_boost
            )
        except Exception as e:
            st.error(f"PDF 미리보기 중 오류 발생: {e}")
    elif uploaded_file:
        st.warning("PDF 미리보기를 표시할 수 없습니다.")

# 왼쪽 컬럼: 채팅 및 설정
with col_left:
    chat_container = st.container(height=500)
    
    new_file_uploaded = uploaded_file and uploaded_file.name != st.session_state.get("last_uploaded_file_name")
    if new_file_uploaded:
        if st.session_state.temp_pdf_path:
            reset_session_state(uploaded_file)
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"📂 새 PDF 파일 '{uploaded_file.name}'이(가) 업로드되었습니다.",
            })
        else:
            st.warning("PDF 파일을 임시로 저장하는 데 실패했습니다. 다시 시도해 주세요.")
            
    # PDF 처리 상태 확인
    if uploaded_file and not st.session_state.pdf_processed and not st.session_state.pdf_processing_error:
        with chat_container:
            with st.spinner("📄 PDF 문서 처리 중... 잠시만 기다려 주세요."):
                qa_chain, documents, embedder, vector_store = process_pdf(
                    uploaded_file,
                    selected_model,
                    st.session_state.temp_pdf_path
                    )
            
    # 채팅 메시지 표시
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # 채팅 입력창
    is_ready_for_input = st.session_state.pdf_processed and not st.session_state.pdf_processing_error
    user_input = st.chat_input(
        "PDF 내용에 대해 질문해보세요.",
        key='user_input',
        disabled=not is_ready_for_input
    )

    # 사용자 입력 처리
    if user_input:
        st.session_state.messages.append({"role": "user",
                                          "content": user_input,
                                          })
        with chat_container:
            with st.chat_message("user"):
                st.write(user_input)

        # 답변 생성
        qa_chain = st.session_state.get("qa_chain")
        if not qa_chain:
            error_message = "❌ QA 체인이 준비되지 않았습니다. PDF 문서를 먼저 성공적으로 처리해야 합니다."
            st.session_state.messages.append({"role": "assistant",
                                              "content": error_message,
                                              })

        if qa_chain:
            with chat_container:
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    message_placeholder.write("▌")
                    try:
                        full_response = ""
                        # 답변 생성
                        logging.info("답변 생성 시작...")
                        stream = qa_chain.stream({
                            "input": user_input
                        })
                        for chunk in stream:
                            answer_part = chunk.get("answer", "")
                            if answer_part:
                                full_response += answer_part
                                message_placeholder.write(full_response + "▌")
                        message_placeholder.write(full_response)
                    except Exception as e:
                        logging.error(f"답변 생성 중 오류 발생: {e}", exc_info=True)
                        full_response = f"❌ 답변 생성 중 오류가 발생했습니다: {e}"
                        message_placeholder.error(full_response)
            st.session_state.messages.append({"role": "assistant",
                                              "content": full_response,
                                              })