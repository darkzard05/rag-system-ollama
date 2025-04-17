import os
os.environ["CHROMA_TELEMETRY"] = "FALSE"
import torch
import time
import tempfile
import subprocess
import logging
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders.parsers import RapidOCRBlobParser
import streamlit as st

from typing import List, Optional
from langchain_core.messages import AIMessage

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
    split_documents.clear()
    create_vector_store.clear()

def prepare_chat_history():
    """이전 대화 기록을 준비합니다."""
    chat_history = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            chat_history.append(AIMessage(content=msg["content"]))
    return chat_history

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
        loader = PyMuPDFLoader(temp_path,
                               extract_tables="markdown",
                               images_inner_format="markdown-img",
                               images_parser=RapidOCRBlobParser())
        docs = loader.load()
        os.remove(temp_path)
        return docs
    except Exception as e:
        logging.error(f"PDF 로드 중 오류 발생: {e}")
        raise ValueError(f"PDF 로드 중 오류 발생: {e}") from e

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
def create_vector_store(_documents, _embedder) -> Optional[Chroma]:
    """문서에서 Chroma 벡터 저장소를 생성하는 함수"""
    logging.info("Chroma 벡터 저장소 생성 중...")
    start_time = time.time()
    try:
        vector_space = Chroma.from_documents(
            documents=_documents,
            embedding=_embedder,
        )
        logging.info(f"Chroma 벡터 저장소 생성 완료 (소요 시간: {time.time() - start_time:.2f}초)")
        return vector_space
    except Exception as e:
        logging.error(f"Chroma 벡터 저장소 생성 중 오류 발생: {e}")
        raise ValueError(f"Chroma 벡터 저장소 생성 중 오류 발생: {e}") from e
    
def process_pdf(uploaded_file, selected_model):
    """PDF 처리 및 QA 체인 생성."""
    try:
        logging.info("PDF 처리 시작...")
        file_bytes = uploaded_file.getvalue()

        logging.info("문서 로딩 중...")
        docs = load_pdf_docs(file_bytes)
        if not docs: raise ValueError("PDF 문서 로딩 실패")

        logging.info("임베딩 모델 로딩 중...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"사용할 장치: {device}")
        embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-m3",
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
            llm = OllamaLLM(model=selected_model)
            if not llm: raise ValueError("LLM 초기화 실패")
            st.session_state.llm = llm
        else:
            raise ValueError("LLM 초기화를 위한 모델 미선택")

        logging.info("QA 체인 생성 중...")
        QA_PROMPT = ChatPromptTemplate.from_messages([
            ("system", ("당신은 주어진 문맥(context)을 바탕으로 질문에 답변하는 AI 어시스턴트입니다.\n"
                        "문맥에서 정보를 찾을 수 없으면, 모른다고 솔직하게 답하세요.\n"
                        "추측하거나 외부 정보를 사용하지 마세요.\n"
                        "답변은 간결하고 명확해야 하며, 항상 한국어로 작성하세요.\n\n"
                        "<context>\n{context}\n</context>")),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
            ])
        combine_chain = create_stuff_documents_chain(st.session_state.llm, QA_PROMPT)
        qa_chain = create_retrieval_chain(
            st.session_state.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={'k': 7, 'fetch_k': 20, 'lambda_mult': 0.5}
                ),
            combine_chain
        )
        st.session_state.qa_chain = qa_chain
        st.session_state.pdf_processed = True
        logging.info("PDF 처리 완료.")
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"✅ PDF 파일 '{uploaded_file.name}'의 문서 처리가 완료되었습니다. 이제 질문할 수 있습니다."
        })
        return qa_chain, documents, embedder, vector_store

    except Exception as e:
        logging.error(f"PDF 처리 중 오류 발생: {e}", exc_info=True)
        st.session_state.pdf_processing_error = str(e)
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"❌ PDF 처리 중 오류가 발생했습니다: {e}"
        })