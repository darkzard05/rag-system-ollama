import os
import torch
import time
import subprocess
import logging
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import streamlit as st
from typing import List, Optional

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
        "temp_pdf_path": None, # 임시 PDF 파일 경로 저장용
        "pdf_is_processing": False # PDF 처리 중 상태 플래그
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
            
def reset_session_state(uploaded_file):
    """세션 상태를 초기화합니다."""
    logging.info("세션 상태 리셋 중...")
    st.session_state.last_uploaded_file_name = uploaded_file.name
    st.session_state.pdf_processed = False
    st.session_state.pdf_processing_error = None
    st.session_state.qa_chain = None
    st.session_state.vector_store = None
    st.session_state.pdf_is_processing = False # PDF 처리 중 상태 플래그 리셋
    # st.session_state.messages = []  # 새 파일이므로 채팅 기록 초기화
    load_pdf_docs.clear()
    split_documents.clear()
    create_vector_store.clear()

@st.cache_data(show_spinner=False)
def get_ollama_models() -> List[str]:
    """Ollama 모델 목록을 가져오는 함수"""
    logging.info("Ollama 모델 목록을 불러오기 시작...")
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        models = [line.split()[0] for line in result.stdout.split("\n")[1:] if line]
        return models
    except Exception as e:
        logging.error(f"Ollama 모델 목록을 불러오는 중 오류 발생: {e}")
        raise ValueError(f"Ollama 모델 목록을 불러오는 중 오류 발생: {e}") from e

@st.cache_resource(show_spinner=False)
def load_pdf_docs(pdf_file_path: str) -> List:
    """PDF 파일을 로드하는 함수"""
    logging.info("PDF 파일 로드 시작...")
    if not os.path.exists(pdf_file_path):
        logging.error(f"PDF 파일 경로가 존재하지 않습니다: {pdf_file_path}")
        raise FileNotFoundError(f"PDF 파일 경로가 존재하지 않습니다: {pdf_file_path}")
    try:
        loader = UnstructuredPDFLoader(
            pdf_file_path,
            )
        docs = loader.lazy_load()
        logging.info(f"PDF 파일 로드 완료")
        return docs
    except Exception as e:
        logging.error(f"PDF 로드 중 오류 발생: {e}")
        raise ValueError(f"PDF 로드 중 오류 발생: {e}") from e

@st.cache_resource(show_spinner=False)
def load_embedding_model() -> HuggingFaceEmbeddings:
    """HuggingFace 임베딩 모델을 로드하고 캐싱합니다."""
    logging.info("임베딩 모델 로딩 시작...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"임베딩 모델용 장치: {device}")
    try:
        embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", # "BAAI/bge-m3", "intfloat/e5-base-v2",
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True, 'device': device},
        )
        logging.info("임베딩 모델 로딩 완료.")
        return embedder
    except Exception as e:
        logging.error(f"임베딩 모델 로딩 중 오류 발생: {e}", exc_info=True)
        raise ValueError(f"임베딩 모델 로딩 실패: {e}") from e

@st.cache_data(show_spinner=False)
def split_documents(_docs: List) -> List:
    """문서를 분할하는 함수"""
    logging.info("문서 분할 시작...")
    start_time = time.time()
    try:
        chunker = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            length_function=len, # 문자 수 기준
            is_separator_regex=False,
            )
        docs = chunker.split_documents(_docs)
        logging.info(f"문서 {len(docs)} 페이지 분할 완료 (소요 시간: {time.time() - start_time:.2f}초)")
        return docs
    except Exception as e:
        logging.error(f"문서 분할 중 오류 발생: {e}")
        raise ValueError(f"문서 분할 중 오류 발생: {e}") from e

@st.cache_resource(show_spinner=False)
def create_vector_store(_documents, _embedder) -> Optional[FAISS]:
    """문서에서 FAISS 벡터 저장소를 생성하는 함수"""
    logging.info("FAISS 벡터 저장소 생성 시작...")
    start_time = time.time()
    try:
        vector_space = FAISS.from_documents(
            documents=_documents,
            embedding=_embedder,
        )
        logging.info(f"FAISS 벡터 저장소 생성 완료 (소요 시간: {time.time() - start_time:.2f}초)")
        return vector_space
    except Exception as e:
        logging.error(f"FAISS 벡터 저장소 생성 중 오류 발생: {e}")
        raise ValueError(f"FAISS 벡터 저장소 생성 중 오류 발생: {e}") from e
    
@st.cache_resource(show_spinner=False)
def load_llm(model_name: str) -> OllamaLLM:
    """선택된 Ollama LLM을 로드하고 캐싱합니다."""
    logging.info(f"Ollama LLM 로딩 시작: {model_name}")
    try:
        llm = OllamaLLM(
            model=model_name,
            )
        logging.info(f"Ollama LLM 로딩 완료: {model_name}")
        return llm
    except Exception as e:
        logging.error(f"Ollama LLM ({model_name}) 로딩 중 오류 발생: {e}", exc_info=True)
        raise ValueError(f"Ollama LLM ({model_name}) 로딩 실패: {e}") from e
    
# QA 프롬프트를 함수 외부에서 정의하여 다른 모듈에서 import 가능하게 함
QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an AI assistant. Your task is to answer questions based *solely* on the provided 'Context'.\n"
        "You MUST respond in the same language as the user's question.\n\n"
        "Context:\n{context}\n\n"
        "Follow these instructions:\n"
        "1. **Internal Thought Process (Mandatory):** Enclose your step-by-step reasoning in `<think>...</think>` tags. Explain how the context leads to your answer, or why it doesn't contain the answer. This is for internal logging and will not be shown to the user.\n"
        "2. **Answer Formulation:** Based *strictly* on the 'Context', construct your answer. It must be clear, detailed, and in the same language as the user's question. Do NOT use any information outside the 'Context'.\n"
        "3. **Handling Missing Information:** If the 'Context' does not provide an answer, respond *only* with a polite statement in the user's language indicating this. For example:\n"
        "   - Korean: '죄송합니다만, 제공된 문서 내용만으로는 요청하신 정보에 대한 답변을 찾을 수 없습니다.'\n"
        "   - English: 'I apologize, but the provided document context does not contain the information needed to answer your question.'\n"
        "   (Adapt this message to the user's language.)"
    )),
    ("human", "Question: {input}")
])

def process_pdf(uploaded_file, selected_model, temp_pdf_path: str):
    """PDF 처리 및 QA 체인 생성."""
    try:
        docs = load_pdf_docs(temp_pdf_path)
        if not docs: raise ValueError("PDF 문서 로딩 실패")

        embedder = load_embedding_model()
        if not embedder: raise ValueError("임베딩 모델 로딩 실패 (캐시)")

        documents = split_documents(docs)
        if not documents: raise ValueError("문서 분할 실패")

        vector_store = create_vector_store(documents, embedder)
        if not vector_store: raise ValueError("벡터 저장소 생성 실패")
        st.session_state.vector_store = vector_store

        if isinstance(selected_model, str):
            llm = load_llm(selected_model)
            if not llm: raise ValueError("LLM 초기화 실패")
            st.session_state.llm = llm
        else:
            raise ValueError("LLM 초기화를 위한 모델 미선택")

        logging.info("QA 체인 생성 시작...")
        # QA_PROMPT 정의를 함수 외부로 이동했으므로 여기서는 사용만 함
        # 전역으로 정의된 QA_PROMPT 사용
        combine_chain = create_stuff_documents_chain(st.session_state.llm, QA_PROMPT)
        qa_chain = create_retrieval_chain(
            st.session_state.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={'k': 5, 'fetch_k': 20, 'lambda_mult': 0.7},
                ),
            combine_chain
        )
        logging.info("QA 체인 생성 완료.")
        st.session_state.qa_chain = qa_chain
        st.session_state.pdf_processed = True
        logging.info("PDF 처리 완료.")
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                f"✅ PDF 파일 '{uploaded_file.name}'의 문서 처리가 완료되었습니다.\n\n"
                "이제 문서 내용에 대해 자유롭게 질문해보세요. 예를 들면 다음과 같습니다:\n\n"
                "- 이 문서의 주요 내용(key points)은 무엇인가요?\n"
                "- 이 문서를 한 문장으로 요약한다면 어떻게 될까요?\n"
                "- 이 문서에서 가장 주목해야 할 부분은 어디인가요?\n"
                "- 이 문서의 내용을 바탕으로 어떤 질문을 할 수 있을까요?\n"
                "- 이 문서에서 다루는 핵심 개념은 무엇인가요?\n"
            )
        })        
        st.session_state.pdf_is_processing = False # 처리 완료 후 플래그 리셋
        st.rerun()

    except Exception as e:
        logging.error(f"PDF 처리 중 오류 발생: {e}", exc_info=True)
        st.session_state.pdf_processing_error = str(e)
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"❌ PDF 처리 중 오류가 발생했습니다: {e}"
        })        
        st.session_state.pdf_is_processing = False # 오류 발생 시 플래그 리셋
        st.session_state.pdf_processed = False # 명시적으로 처리 안됨 상태로 설정
        st.session_state.qa_chain = None # QA 체인도 초기화
        st.rerun() # 오류 메시지 표시를 위해 rerun