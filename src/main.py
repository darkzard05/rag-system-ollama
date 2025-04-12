import torch
torch.classes.__path__ = []
import tempfile
import os
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import logging
from utils import (
    init_session_state,
    reset_session_state,
    prepare_chat_history,
    generate_example_questions,
    get_ollama_models,
    load_pdf_docs,
    get_embedder,
    split_documents,
    create_vector_store,
    init_llm,
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

    if selected_model and selected_model != st.session_state.last_selected_model:
        st.session_state.last_selected_model = selected_model
        if st.session_state.get("llm"):
            try:
                st.session_state.llm = init_llm(selected_model)
                if st.session_state.get("vector_store"):
                    QA_PROMPT = ChatPromptTemplate.from_messages([
                        ("system", "당신은 주어진 문맥(context) 내용을 바탕으로 질문에 답변하는 AI 어시스턴트입니다. 문맥에서 정보를 찾을 수 없으면, 모른다고 솔직하게 답하세요. 추측하거나 외부 정보를 사용하지 마세요. 항상 한국어로 답변해주세요.\n\n<context>\n{context}\n</context>"),
                        MessagesPlaceholder(variable_name="chat_history"),
                        ("human", "{input}")
                    ])
                    combine_chain = create_stuff_documents_chain(st.session_state.llm, QA_PROMPT)
                    st.session_state.qa_chain = create_retrieval_chain(
                        st.session_state.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
                        combine_chain
                    )
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"🛠️ 모델이 {selected_model}(으)로 변경되었고 QA 체인이 업데이트되었습니다."
                    })
                else:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"🛠️ 모델이 {selected_model}(으)로 변경되었습니다. PDF를 업로드하면 해당 모델로 처리됩니다."
                    })
                st.rerun()
            except Exception as e:
                st.error(f"LLM 또는 QA 체인 업데이트 중 오류 발생: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠️ 모델 {selected_model} 변경 중 오류 발생: {e}"
                })
                st.rerun()

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
            try:
                logging.info("PDF 처리 시작...")
                file_bytes = uploaded_file.getvalue()

                logging.info("문서 로딩 중...")
                docs = load_pdf_docs(file_bytes)
                if not docs: raise ValueError("PDF 문서 로딩 실패")

                logging.info("임베딩 모델 로딩 중...")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                embedder = get_embedder(model_name="BAAI/bge-m3",
                                        model_kwargs={'device': device},
                                        encode_kwargs={'normalize_embeddings': True, 'device': device})
                if not embedder: raise ValueError("임베딩 모델 로딩 실패")

                logging.info("문서 분할 중...")
                documents = split_documents(docs, embedder)
                if not documents: raise ValueError("문서 분할 실패")

                logging.info("벡터 저장소 생성 중...")
                vector_store = create_vector_store(documents, embedder)
                if not vector_store: raise ValueError("벡터 저장소 생성 실패")
                st.session_state.vector_store = vector_store

                logging.info("LLM 초기화 중...")
                if isinstance(selected_model, str):
                    llm = init_llm(selected_model)
                    if not llm: raise ValueError("LLM 초기화 실패")
                    st.session_state.llm = llm
                else:
                    raise ValueError("LLM 초기화를 위한 모델 미선택")

                logging.info("QA 체인 생성 중...")
                QA_PROMPT = ChatPromptTemplate.from_messages([
                    ("system", "당신은 주어진 문맥(context) 내용을 바탕으로 질문에 답변하는 AI 어시스턴트입니다. 문맥에서 정보를 찾을 수 없으면, 모른다고 솔직하게 답하세요. 추측하거나 외부 정보를 사용하지 마세요. 항상 한국어로 답변해주세요.\n\n<context>\n{context}\n</context>"),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}")
                ])
                combine_chain = create_stuff_documents_chain(st.session_state.llm, QA_PROMPT)
                qa_chain = create_retrieval_chain(
                    st.session_state.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10}),
                    combine_chain
                )
                st.session_state.qa_chain = qa_chain
                st.session_state.pdf_processed = True
                logging.info("PDF 처리 완료.")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"✅ PDF 파일 '{uploaded_file.name}'의 문서 처리가 완료되었습니다. 이제 질문할 수 있습니다."
                })

                generate_example_questions()
                st.rerun()

            except Exception as e:
                logging.error(f"PDF 처리 중 오류 발생: {e}", exc_info=True)
                st.session_state.pdf_processing_error = str(e)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ PDF 처리 중 오류가 발생했습니다: {e}"
                })
                st.rerun()

    # 채팅 컨테이너 및 메시지 표시
    chat_container = st.container(height=500, border=True)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    is_ready_for_input = st.session_state.pdf_processed and not st.session_state.pdf_processing_error

    user_input = st.chat_input(
        "PDF 내용에 대해 질문해보세요.",
        key='user_input',
        disabled=not is_ready_for_input
    )

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with chat_container:
            with st.chat_message("user"):
                st.write(user_input)

        qa_chain = st.session_state.get("qa_chain")
        if not qa_chain:
            error_message = "❌ QA 체인이 준비되지 않았습니다. PDF 문서를 먼저 성공적으로 처리해야 합니다."
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            with chat_container:
                with st.chat_message("assistant"):
                    st.warning(error_message)

        if qa_chain:
            with chat_container:
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    message_placeholder.write("▌")
                    try:
                        chat_history = prepare_chat_history()
                        full_response = ""
                        stream = qa_chain.stream({
                            "input": user_input,
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