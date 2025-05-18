import os
import torch
import time
import subprocess
import logging
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
        "processing_step": None,
        "needs_rerun": False  # 리런 필요 여부를 추적하는 새로운 상태
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
        
        # 캐시된 함수들 초기화
        load_pdf_docs.clear()
        split_documents.clear()
        create_vector_store.clear()
        cls.request_rerun()
    
    @classmethod
    def add_message(cls, role: str, content: str, replace_last: bool = False):
        """메시지 추가 또는 마지막 메시지 교체"""
        if replace_last and st.session_state.messages:
            st.session_state.messages[-1] = {"role": role, "content": content}
        else:
            st.session_state.messages.append({"role": role, "content": content})
        cls.request_rerun()
    
    @classmethod
    def update_progress(cls, step: str, message: str):
        """처리 단계 업데이트 및 진행 상황 메시지 표시"""
        st.session_state.processing_step = step
        cls.add_message("assistant", f"🔄 {message}", replace_last=True)
    
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
    def request_rerun(cls):
        """리런이 필요함을 표시"""
        st.session_state.needs_rerun = True

    @classmethod
    def check_and_clear_rerun(cls):
        """리런이 필요한지 확인하고 상태를 초기화"""
        needs_rerun = st.session_state.get("needs_rerun", False)
        st.session_state.needs_rerun = False
        return needs_rerun

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
        start_time = time.time()
        loader = PyMuPDFLoader(
            pdf_file_path,
            )
        docs = loader.load()
        logging.info(f"PDF 파일 로드 완료 (소요 시간: {time.time() - start_time:.2f}초)")
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
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={
                'device': device,
            },
            encode_kwargs={
                'normalize_embeddings': True,
                'device': device,
            },
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
            chunk_size=1000,  # 청크 크기 최적화 (더 작은 크기로 조정)
            chunk_overlap=200,  # 오버랩 크기 증가로 문맥 유지 강화
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],  # 분할 우선순위
            is_separator_regex=False,
            add_start_index=True,  # 시작 인덱스 추가로 추적성 향상
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

@st.cache_resource(show_spinner=False, ttl=3600)
def process_pdf(uploaded_file, selected_model: str, temp_pdf_path: str):
    """PDF 처리 및 QA 체인 생성."""
    try:
        # 초기 진행 상태 메시지
        SessionManager.update_progress("start", "PDF 처리를 시작합니다...")

        # 1. PDF 문서 로드
        SessionManager.update_progress("loading", "PDF 문서를 로드하고 있습니다...")
        docs = load_pdf_docs(temp_pdf_path)
        if not docs:
            raise ValueError("PDF 문서를 로드할 수 없습니다.")

        # 2. 임베딩 모델 로드
        SessionManager.update_progress("embedding", "임베딩 모델을 초기화하고 있습니다...")
        embedder = load_embedding_model()
        if not embedder:
            raise ValueError("임베딩 모델을 로드할 수 없습니다.")

        # 3. 문서 분할
        SessionManager.update_progress("splitting", "문서를 분할하고 있습니다...")
        documents = split_documents(docs)
        if not documents:
            raise ValueError("문서를 분할할 수 없습니다.")

        # 4. 벡터 저장소 생성
        SessionManager.update_progress("vectorizing", "벡터 저장소를 생성하고 있습니다...")
        vector_store = create_vector_store(documents, embedder)
        if not vector_store:
            raise ValueError("벡터 저장소를 생성할 수 없습니다.")
        st.session_state.vector_store = vector_store

        # 5. LLM 초기화
        SessionManager.update_progress("llm_init", f"{selected_model} 모델을 초기화하고 있습니다...")
        if not isinstance(selected_model, str):
            raise ValueError("유효하지 않은 모델명입니다.")
            
        llm = load_llm(selected_model)
        if not llm:
            raise ValueError("LLM을 초기화할 수 없습니다.")
        st.session_state.llm = llm

        # 6. QA 체인 생성
        SessionManager.update_progress("qa_chain", "QA 체인을 생성하고 있습니다...")
        logging.info("QA 체인 생성 시작...")
        combine_chain = create_stuff_documents_chain(
            st.session_state.llm,
            QA_PROMPT
        )
        
        retriever = st.session_state.vector_store.as_retriever(
            search_type=RETRIEVER_CONFIG['search_type'],
            search_kwargs=RETRIEVER_CONFIG['search_kwargs']
        )
        
        qa_chain = create_retrieval_chain(retriever, combine_chain)
        if not qa_chain:
            raise ValueError("QA 체인을 생성할 수 없습니다.")
            
        st.session_state.qa_chain = qa_chain
        st.session_state.pdf_processed = True
        logging.info("PDF 처리 완료.")
        
        # 성공 메시지 생성
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
        logging.error(f"PDF 처리 중 오류 발생: {e}", exc_info=True)
        st.session_state.pdf_processing_error = str(e)
        error_message = f"❌ PDF 처리 중 오류가 발생했습니다: {e}"
        SessionManager.add_message("assistant", error_message)
        st.session_state.pdf_processed = False
        st.session_state.qa_chain = None
    finally:
        st.session_state.pdf_is_processing = False
        st.session_state.processing_step = None
        SessionManager.request_rerun()