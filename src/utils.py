import os
import torch
import time
import subprocess
import logging
import functools
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import streamlit as st
from typing import List, Optional, Dict

# 리트리버 설정 상수
RETRIEVER_CONFIG: Dict = {
    'search_type': "mmr",
    'search_kwargs': {
        'k': 5,           # 검색 결과 수 최적화
        'fetch_k': 20,    # 후보 수 증가
        'lambda_mult': 0.8 # MMR 다양성 가중치 증가
    }
}

class SessionManager:
    """세션 상태를 관리하는 클래스"""
    
    # 세션 상태의 기본값을 클래스 변수로 정의
    DEFAULT_SESSION_STATE = {
        "messages": [],
        "last_selected_model": None,
        "last_uploaded_file_name": None,
        "pdf_processed": False,
        "pdf_processing_error": None,
        "qa_chain": None,
        "vector_store": None,
        "llm": None,
        "temp_pdf_path": None,
        "pdf_is_processing": False,
        "processing_step": None
    }
    
    @classmethod
    def init_session(cls):
        """세션 상태 초기화 - 한 번만 실행되어야 함"""
        if not st.session_state.get("_initialized", False):
            logging.info("세션 상태 초기화 중...")
            for key, value in cls.DEFAULT_SESSION_STATE.items():
                if key not in st.session_state:
                    st.session_state[key] = value
            st.session_state._initialized = True
    
    @classmethod
    def reset_session_state(cls, keys=None):
        """지정된 키들의 세션 상태를 기본값으로 리셋"""
        keys_to_reset = keys if keys is not None else cls.DEFAULT_SESSION_STATE.keys()
        for key in keys_to_reset:
            if key in cls.DEFAULT_SESSION_STATE:
                st.session_state[key] = cls.DEFAULT_SESSION_STATE[key]
    
    @classmethod
    def reset_for_new_file(cls, uploaded_file):
        """새 파일 업로드시 세션 상태 리셋"""
        logging.info("새 파일 업로드로 인한 세션 상태 리셋 중...")
        file_related_keys = [
            "last_uploaded_file_name",
            "pdf_processed",
            "pdf_processing_error",
            "qa_chain",
            "vector_store",
            "pdf_is_processing",
            "processing_step",
            "messages"
        ]
        cls.reset_session_state(file_related_keys)
        st.session_state.last_uploaded_file_name = uploaded_file.name
        
        # Streamlit 캐시 초기화
        st.cache_data.clear()
        st.cache_resource.clear()
    
    @classmethod
    def add_message(cls, role: str, content: str):
        """메시지 추가"""
        if not st.session_state.get("messages"):
            st.session_state.messages = []
        st.session_state.messages.append({"role": role, "content": content})
    
    @classmethod
    def update_progress(cls, step: str, message: str):
        """처리 단계 업데이트 및 진행 상황 메시지 표시"""
        st.session_state.processing_step = step
        cls.add_message("assistant", f"🔄 {message}")
    
    @staticmethod
    def is_ready_for_chat():
        """채팅 준비 상태 확인"""
        return (st.session_state.get("pdf_processed") and 
                not st.session_state.get("pdf_processing_error") and 
                st.session_state.get("qa_chain") is not None)
    
    @classmethod
    def update_model(cls, new_model: str):
        """모델 업데이트"""
        old_model = st.session_state.get("last_selected_model", "N/A")
        model_related_keys = ["last_selected_model", "llm", "qa_chain"]
        cls.reset_session_state(model_related_keys)
        st.session_state.last_selected_model = new_model
        
        cls.add_message(
            "assistant", 
            f"🔄 모델을 {new_model}로 변경합니다."
        )
        return old_model

    @classmethod
    def handle_error(cls, error: Exception, error_context: str, affected_states: list = None):
        """에러 처리 및 상태 업데이트"""
        error_msg = f"{error_context}: {str(error)}"
        logging.error(error_msg, exc_info=True)
        
        if affected_states:
            cls.reset_session_state(affected_states)
            
        cls.add_message("assistant", f"❌ {error_msg}")
        return error_msg
    
    @classmethod
    def set_error_state(cls, error_message: str, error_context: str = None):
        """에러 상태 설정"""
        st.session_state.pdf_processing_error = error_message
        if error_context:
            logging.error(f"{error_context}: {error_message}")
        cls.add_message("assistant", f"❌ {error_message}")
    
    @classmethod
    def clear_error_state(cls):
        """에러 상태 초기화"""
        st.session_state.pdf_processing_error = None

# 로깅 데코레이터 수정
def log_operation(operation_name):
    def decorator(func):
        @functools.wraps(func)  # 함수 메타데이터 보존
        def wrapper(*args, **kwargs):
            logging.info(f"{operation_name} 시작...")
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                logging.info(f"{operation_name} 완료 (소요 시간: {time.time() - start_time:.2f}초)")
                return result
            except Exception as e:
                logging.error(f"{operation_name} 중 오류 발생: {e}", exc_info=True)
                raise
        return wrapper
    return decorator

# Streamlit 캐시 데코레이터를 항상 바깥쪽에 배치
@st.cache_data(show_spinner=False)
@log_operation("Ollama 모델 목록 불러오기")
def get_ollama_models() -> List[str]:
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
    return [line.split()[0] for line in result.stdout.split("\n")[1:] if line]

@st.cache_resource(show_spinner=False)
@log_operation("PDF 파일 로드")
def load_pdf_docs(pdf_file_path: str) -> List:
    if not os.path.exists(pdf_file_path):
        raise FileNotFoundError(f"PDF 파일 경로가 존재하지 않습니다: {pdf_file_path}")
    loader = PyMuPDFLoader(pdf_file_path)
    return loader.load()

@st.cache_resource(show_spinner=False)
@log_operation("임베딩 모델 로딩")
def load_embedding_model() -> HuggingFaceEmbeddings:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"임베딩 모델용 장치: {device}")
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': device, 'trust_remote_code': True},
        encode_kwargs={'device': device, 'batch_size': 32},
    )
    return embedder

@st.cache_data(show_spinner=False)
@log_operation("문서 분할")
def split_documents(_docs: List) -> List:
    chunker = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        is_separator_regex=False,
        add_start_index=True,
    )
    return chunker.split_documents(_docs)

@st.cache_resource(show_spinner=False)
@log_operation("FAISS 벡터 저장소 생성")
def create_vector_store(_documents, _embedder) -> Optional[FAISS]:
    return FAISS.from_documents(
        documents=_documents,
        embedding=_embedder,
    )

@st.cache_resource(show_spinner=False)
@log_operation("Ollama LLM 로딩")
def load_llm(model_name: str) -> OllamaLLM:
    return OllamaLLM(model=model_name)

# QA 프롬프트를 함수 외부에서 정의하여 다른 모듈에서 import 가능하게 함
QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an AI assistant. Your primary task is to answer questions based *solely* on the provided 'Context'.\n\n"
        "**CRITICAL: You MUST respond in the EXACT same language as the user's question.** This is the most important instruction.\n\n"
        "Context:\n"
        "{context}\n"
        "Follow these instructions carefully:\n"
        "1. **Language of Response:**\n"
        "   - ALWAYS use the same language as the user's question for your entire response.\n"
        "   - For example, if the question is in Korean, your answer MUST be in Korean. If the question is in English, your answer MUST be in English.\n\n"
        "2. **Answer Formulation:**\n"
        "   - Construct your answer based *strictly* on the information found within the 'Context'.\n"
        "   - Your answer should be clear and detailed.\n"
        "   - Do NOT use any external knowledge or information not present in the 'Context'.\n\n"
        "3. **Handling Missing Information:**\n"
        "   - If the 'Context' does not contain the information to answer the question, you MUST state (in the same language as the question) that the information is not available in the provided document.\n"
        "   - Do not invent an answer or use external knowledge."
        )),
    ("human", "Question: {input}")
    ])

def process_pdf(uploaded_file, selected_model: str, temp_pdf_path: str):
    """PDF 처리 및 QA 체인 생성."""
    try:
        # 상태 초기화
        st.session_state.pdf_is_processing = True
        st.session_state.pdf_processed = False
        st.session_state.qa_chain = None
        
        # 각 단계 처리
        docs = load_pdf_docs(temp_pdf_path)
        embedder = load_embedding_model()
        documents = split_documents(docs)
        vector_store = create_vector_store(documents, embedder)
        llm = load_llm(selected_model)
        
        # QA 체인 생성
        combine_chain = create_stuff_documents_chain(llm, QA_PROMPT)
        retriever = vector_store.as_retriever(
            search_type=RETRIEVER_CONFIG['search_type'],
            search_kwargs=RETRIEVER_CONFIG['search_kwargs']
        )
        qa_chain = create_retrieval_chain(retriever, combine_chain)
        
        # 세션 상태 한번에 업데이트
        st.session_state.update({
            'vector_store': vector_store,
            'llm': llm,
            'qa_chain': qa_chain,
            'pdf_processed': True,
            'pdf_processing_error': None
        })
        
        # 성공 메시지
        success_message = (
            f"✅ '{uploaded_file.name}' 문서 처리가 완료되었습니다.\n\n"
            "다음과 같은 질문들을 해보세요:\n\n"
            "[문서 전체 이해하기]\n"
            "- 이 문서를 한 문단으로 요약해주세요\n"
            "- 이 문서의 주요 주장과 근거를 설명해주세요\n"
            "- 이 문서의 핵심 용어 3가지를 설명해주세요\n\n"
            "[세부 내용 파악하기]\n"
            "- 이 문서가 해결하고자 하는 문제는 무엇인가요?\n"
            "- 문서에서 제시된 해결책이나 제안은 무엇인가요?\n"
            "- 이 연구의 한계점이나 향후 연구 방향은 무엇인가요?\n\n"
            "자유롭게 문서의 내용에 대해 질문해보세요."
        )
        SessionManager.add_message("assistant", success_message)

    except Exception as e:
        SessionManager.handle_error(
            error=e,
            error_context="PDF 처리",
            affected_states=['pdf_processed', 'qa_chain', 'vector_store', 'llm']
        )
        raise
    finally:
        st.session_state.pdf_is_processing = False
        st.session_state.processing_step = None
    
    st.rerun()