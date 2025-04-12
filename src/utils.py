import os
import time
import tempfile
import subprocess
import logging
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st

from typing import List, Optional, Dict, Any


def init_session_state():
    """세션 상태 초기화 함수"""
    logging.info("세션 상태 초기화 중...")
    defaults = {
        "messages": [],
        "last_selected_model": None,
        "last_uploaded_file_name": None,
        "pdf_processed": False,
        "pdf_processing_error": None,
        "qa_chain": None,
        "vector_store": None,
        "llm": None,
        "temp_pdf_path": None # 임시 PDF 파일 경로 저장용
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
            
def reset_session_state(uploaded_file):
    """세션 상태를 초기화합니다."""
    st.session_state.last_uploaded_file_name = uploaded_file.name
    st.session_state.pdf_processed = False
    st.session_state.pdf_processing_error = None
    st.session_state.qa_chain = None
    st.session_state.vector_store = None
    st.session_state.messages = []  # 새 파일이므로 채팅 기록 초기화
    load_pdf_docs.clear()
    get_embedder.clear()
    split_documents.clear()
    create_vector_store.clear()
    init_llm.clear()

def prepare_chat_history():
    """이전 대화 기록을 준비합니다."""
    chat_history = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            chat_history.append(AIMessage(content=msg["content"]))
    return chat_history

def generate_example_questions():
    """예시 질문을 생성합니다."""
    try:
        with st.spinner("💡 문서 기반 예시 질문 생성 중..."):
            logging.info("예시 질문 생성 시작...")
            example_question_prompt = "이 문서의 내용을 기반으로 사용자가 궁금해할 만한 흥미로운 질문 5가지를 한국어로 만들어 주세요. 질문만 목록 형태로 제시해 주세요."
            chat_history = prepare_chat_history()
            response = st.session_state.qa_chain.invoke({
                "input": example_question_prompt,
                "chat_history": chat_history
            })
            example_questions = response.get("answer", "⚠️ 예시 질문 생성 실패")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"💡 다음은 이 문서에 대해 질문해 볼 수 있는 예시입니다:\n\n{example_questions}"
            })
            logging.info("예시 질문 생성 완료.")
    except Exception as e:
        logging.warning(f"예시 질문 생성 중 오류 발생: {e}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": "⚠️ 예시 질문 생성 중 오류가 발생했습니다."
        })

@st.cache_data(show_spinner=False)
def get_ollama_models() -> List[str]:
    """Ollama 모델 목록을 가져오는 함수"""
    logging.info("Ollama 모델 목록을 불러오는 중...")
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        models = [line.split()[0] for line in result.stdout.split("\n")[1:] if line]
        return models
    except Exception as e:
        logging.error(f"Ollama 모델 목록을 불러오는 중 오류 발생: {e}")
        raise ValueError(f"Ollama 모델 목록을 불러오는 중 오류 발생: {e}") from e

@st.cache_data(show_spinner=False)
def load_pdf_docs(file_path) -> List:
    """PDF 파일을 로드하는 함수"""
    logging.info("PDF 파일 로드 중...")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_path)
            temp_path = tmp_file.name
        loader = PyMuPDFLoader(temp_path)
        docs = loader.load()
        os.remove(temp_path)
        return docs
    except Exception as e:
        logging.error(f"PDF 로드 중 오류 발생: {e}")
        raise ValueError(f"PDF 로드 중 오류 발생: {e}") from e

@st.cache_resource(show_spinner=False)
def get_embedder(model_name, model_kwargs=None, encode_kwargs=None) -> HuggingFaceEmbeddings:
    """HuggingFaceEmbeddings 모델을 초기화하는 함수"""
    try:
        return HuggingFaceEmbeddings(model_name=model_name,
                                    model_kwargs=model_kwargs,
                                    encode_kwargs=encode_kwargs)
    except Exception as e:
        logging.error(f"임베더 초기화 중 오류 발생: {e}")
        raise ValueError(f"임베더 초기화 중 오류 발생: {e}") from e

@st.cache_data(show_spinner=False)
def split_documents(_docs: List, _embedder) -> List:
    """문서를 분할하는 함수"""
    logging.info("문서 분할 시작...")
    start_time = time.time()
    try:
        chunker = SemanticChunker(_embedder)
        docs = chunker.split_documents(_docs)
        logging.info(f"문서 {len(docs)} 페이지 분할 완료 (소요 시간: {time.time() - start_time:.2f}초)")
        return docs
    except Exception as e:
        logging.error(f"문서 분할 중 오류 발생: {e}")
        raise ValueError(f"문서 분할 중 오류 발생: {e}") from e

@st.cache_resource(show_spinner=False)
def create_vector_store(_documents, _embedder) -> Optional[FAISS]:
    """문서에서 벡터 저장소를 생성하는 함수"""
    logging.info("벡터 저장소 생성 중...")
    start_time = time.time()
    try:
        vector_space = FAISS.from_documents(_documents, _embedder)
        logging.info(f"벡터 저장소 생성 완료 (소요 시간: {time.time() - start_time:.2f}초)")
        return vector_space
    except Exception as e:
        logging.error(f"벡터 저장소 생성 중 오류 발생: {e}")
        raise ValueError(f"벡터 저장소 생성 중 오류 발생: {e}") from e

@st.cache_resource(show_spinner=False)
def init_llm(model_name) -> Optional[OllamaLLM]:
    """LLM을 초기화하는 함수"""
    logging.info("LLM 초기화 중...")
    try:
        return OllamaLLM(model=model_name, additional_settings={"output_format": "plain_text"})
    except Exception as e:
        logging.error(f"LLM 초기화 중 오류 발생: {e}")
        raise ValueError(f"LLM 초기화 중 오류 발생: {e}") from e