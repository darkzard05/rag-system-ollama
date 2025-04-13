import torch
torch.classes.__path__ = []
import tempfile
import os
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import logging
from utils import (
    init_session_state,
    reset_session_state,
    prepare_chat_history,
    get_ollama_models,
    process_pdf,
)

# 페이지 설정
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📄 RAG Chatbot with Ollama LLM")

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# 세션 상태 초기화
init_session_state()

# 사이드바 설정
with st.sidebar:
    st.header("Settings")
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
    st.header("📄 PDF Viewer Settings")
    resolution_boost = st.slider(label="Resolution boost", min_value=1, max_value=10, value=1)
    width = st.slider(label="PDF width", min_value=100, max_value=1000, value=1000)
    height = st.slider(label="PDF height", min_value=-1, max_value=10000, value=1000)

# 레이아웃 설정
col_left, col_right = st.columns([1, 1])

# 오른쪽 컬럼: PDF 미리보기
with col_right:
    st.header("📄 PDF Preview")
    with st.container():  # 컨테이너 추가
        if uploaded_file and uploaded_file.name != st.session_state.get("last_uploaded_file_name"):
            if st.session_state.temp_pdf_path and os.path.exists(st.session_state.temp_pdf_path):
                try:
                    os.remove(st.session_state.temp_pdf_path)
                except Exception as e:
                    logging.warning(f"이전 임시 PDF 파일 삭제 실패: {e}")
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    st.session_state.temp_pdf_path = tmp.name
            except Exception as e:
                st.error(f"임시 PDF 파일 생성 실패: {e}")
                st.session_state.temp_pdf_path = None

        if st.session_state.temp_pdf_path and os.path.exists(st.session_state.temp_pdf_path):
            try:
                pdf_viewer(
                    input=st.session_state.temp_pdf_path,
                    width=width,
                    height=height,
                    key=f'pdf_viewer_{st.session_state.last_uploaded_file_name}',
                    resolution_boost=resolution_boost
                )
            except Exception as e:
                st.error(f"PDF 미리보기 중 오류 발생: {e}")
        elif uploaded_file:
            st.warning("PDF 미리보기를 표시할 수 없습니다.")

# 왼쪽 컬럼: 채팅 및 설정
with col_left:
    st.header("💬 Chat")
    chat_container = st.container(height=500, border=True)  # 채팅 컨테이너 추가
    with chat_container:
        # 채팅 메시지 표시
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # 문서 처리 상태 메시지 추가
        new_file_uploaded = uploaded_file and uploaded_file.name != st.session_state.get("last_uploaded_file_name")
        if new_file_uploaded:
            reset_session_state(uploaded_file)
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"📂 새 PDF 파일 '{uploaded_file.name}'이(가) 업로드되었습니다. 문서를 처리합니다..."
            })
            st.rerun()

        if uploaded_file and not st.session_state.pdf_processed and not st.session_state.pdf_processing_error:
            with st.spinner("📄 PDF 문서 처리 중... 잠시만 기다려 주세요."):
                process_pdf(uploaded_file, selected_model)

                # qa_chain 상태 확인
                if not st.session_state.qa_chain:
                    logging.error("QA 체인이 초기화되지 않았습니다.")
                    with chat_container:
                        with st.chat_message("assistant"):
                            st.error("⚠️ QA 체인이 초기화되지 않아 예시 질문을 생성할 수 없습니다.")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "⚠️ QA 체인이 초기화되지 않아 예시 질문을 생성할 수 없습니다."
                    })
                else:
                    # 예시 질문 생성 및 컨테이너 안에 추가
                    with chat_container:
                        with st.chat_message("assistant"):
                            example_placeholder = st.empty()
                            example_placeholder.write("▌")
                            try:
                                logging.info("예시 질문 생성 시작...")
                                example_question_prompt = (
                                    "문서의 핵심 내용을 바탕으로 사용자가 궁금해할 만한 질문 5개를 생성하세요.\n"
                                    "반드시 한국어로 답변하세요."
                                )
                                stream = st.session_state.qa_chain.stream({
                                    "input": example_question_prompt,
                                    "chat_history": [],
                                })
                                example_questions = ""
                                for chunk in stream:
                                    answer_part = chunk.get("answer", "")
                                    if answer_part:
                                        example_questions += answer_part
                                        example_placeholder.write(example_questions + "▌")
                                example_placeholder.write(example_questions)  # 최종 출력
                                logging.info("예시 질문 생성 완료.")
                            except Exception as e:
                                logging.warning(f"예시 질문 생성 중 오류 발생: {e}")
                                example_placeholder.error("⚠️ 예시 질문 생성 중 오류가 발생했습니다.")
                                example_questions = "⚠️ 예시 질문 생성 중 오류가 발생했습니다."

                    # 예시 질문을 세션 메시지에 추가
                    st.session_state.messages.append({"role": "assistant", "content": example_questions})

    # 채팅 입력창을 컨테이너 하단에 고정
    is_ready_for_input = st.session_state.pdf_processed and not st.session_state.pdf_processing_error
    user_input = st.chat_input(
        "PDF 내용에 대해 질문해보세요.",
        key='user_input',
        disabled=not is_ready_for_input
    )

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with chat_container:  # 입력 메시지도 컨테이너 안에 추가
            with st.chat_message("user"):
                st.write(user_input)

        qa_chain = st.session_state.get("qa_chain")
        if not qa_chain:
            error_message = "❌ QA 체인이 준비되지 않았습니다. PDF 문서를 먼저 성공적으로 처리해야 합니다."
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            with chat_container:  # 오류 메시지도 컨테이너 안에 추가
                with st.chat_message("assistant"):
                    st.warning(error_message)

        if qa_chain:
            with chat_container:  # 답변도 컨테이너 안에 추가
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    message_placeholder.write("▌")
                    try:
                        chat_history = prepare_chat_history()
                        full_response = ""
                        # 입력 프롬프트에 한국어로 답변 요청 추가
                        stream = qa_chain.stream({
                            "input": f"{user_input}\n\n항상 한국어로 답변하세요.",
                            "chat_history": chat_history
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
            st.session_state.messages.append({"role": "assistant", "content": full_response})