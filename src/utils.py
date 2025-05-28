import os
import torch
import time
import subprocess
import logging
import functools
from typing import List, Optional, Dict
import streamlit as st

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# 모델 및 설정 상수
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# 리트리버 설정 상수
RETRIEVER_CONFIG: Dict = {
    'search_type': "similarity",
    'search_kwargs': {
        'k': 5,
    }
}

# 텍스트 분할 설정
TEXT_SPLITTER_CONFIG: Dict = {
    'chunk_size': 4000,
    'chunk_overlap': 200,
}

class SessionManager:
    """세션 상태를 관리하는 클래스"""
      # 세션 상태의 기본값을 클래스 변수로 정의
    DEFAULT_SESSION_STATE = {
        # 기본 상태
        "messages": [],
        "last_selected_model": None,
        "last_uploaded_file_name": None,
        "last_model_change_message": None,
        
        # PDF 처리 상태
        "pdf_processed": False,
        "pdf_processing_error": None,
        "pdf_is_processing": False,
        "temp_pdf_path": None,
        "processing_step": None,
        
        # 문서 처리 관련
        "processed_document_splits": None,
        "source_documents": {},
        "current_file_path": None,
        
        # LLM 및 검색 관련
        "qa_chain": None,
        "vector_store": None,
        "llm": None,
        "bm25_retriever": None,
        
        # 캐시 관련
        "_faiss_index": None,
        "_pdf_text_cache": None,
        "last_retriever_key": None,
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
        
        # 보존할 상태 저장
        preserved_states = {
            "last_model_change_message": st.session_state.get("last_model_change_message"),
            "last_selected_model": st.session_state.get("last_selected_model"),
            "_initialized": st.session_state.get("_initialized", False)
        }
        
        # Streamlit 캐시 초기화
        st.cache_resource.clear()
        st.cache_data.clear()
        
        # 모든 세션 상태 초기화 (보존할 상태 제외)
        exclude_keys = ["last_selected_model", "_initialized"]
        all_keys = [key for key in st.session_state.keys() if key not in exclude_keys]
        cls.reset_session_state(all_keys)
        
        # 새 파일 정보 설정
        st.session_state.last_uploaded_file_name = uploaded_file.name
        st.session_state.current_file_path = None  # 새로운 처리 과정에서 설정됨
        
        # 보존된 상태 복원
        for key, value in preserved_states.items():
            if value is not None and key != "_initialized":
                st.session_state[key] = value
                if key == "last_model_change_message":
                    cls.add_message("assistant", value)
        
        # 초기화 완료 로깅
        logging.info(f"세션 상태 초기화 완료 - 새 파일: {uploaded_file.name}")
    
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
            f"🔄 모델을 {new_model}로 변경했습니다."
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

    @classmethod
    def clear_caches(cls):
        """모든 캐시 초기화"""
        # Streamlit 캐시 초기화
        st.cache_resource.clear()
        st.cache_data.clear()
        
        # 캐시 관련 세션 상태 초기화
        cache_keys = [
            "vector_store",
            "bm25_retriever",
            "_faiss_index",
            "_pdf_text_cache",
            "last_retriever_key",
            "processed_document_splits"
        ]
        for key in cache_keys:
            if key in st.session_state:
                del st.session_state[key]
        logging.info("모든 캐시가 초기화되었습니다.")
    
    @classmethod
    def get_file_specific_cache_key(cls, base_key: str) -> str:
        """파일별 고유 캐시 키 생성"""
        current_file = st.session_state.get('current_file_path', '')
        current_model = st.session_state.get('last_selected_model', '')
        return f"{base_key}_{current_file}_{current_model}"

# 로깅 데코레이터 수정
def log_operation(operation_name):
    def decorator(func):
        @functools.wraps(func)
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

@st.cache_data(show_spinner=False)
@log_operation("Ollama 모델 목록 불러오기")
def get_ollama_models() -> List[str]:
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
    return [line.split()[0] for line in result.stdout.split("\n")[1:] if line]

@st.cache_resource(show_spinner=False, ttl=600)  # 10분 후 캐시 만료
@log_operation("PDF 파일 로드")
def load_pdf_docs(pdf_file_path: str) -> List:
    if not os.path.exists(pdf_file_path):
        raise FileNotFoundError(f"PDF 파일 경로가 존재하지 않습니다: {pdf_file_path}")
    loader = PyMuPDFLoader(
        pdf_file_path,
        mode="page",
        )
    return loader.load()

@st.cache_resource(show_spinner=False)
@log_operation("임베딩 모델 로딩")
def load_embedding_model() -> HuggingFaceEmbeddings:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"임베딩 모델용 장치: {device}")
      # GPU 메모리 최적화
    if device == "cuda":
        # CUDA 캐시 정리
        torch.cuda.empty_cache()
        # 메모리 할당자 설정
        torch.backends.cudnn.benchmark = True
        # GPU 메모리 할당자 최적화
        torch.backends.cuda.max_split_size_mb = 512
    
    # 모델 설정 (SentenceTransformer는 torch_dtype를 직접 지원하지 않음)
    model_kwargs = {
        "device": device,
        "trust_remote_code": False,
    }
    
    # 인코딩 설정 최적화
    encode_kwargs = {
        "device": device,
        "batch_size": 128,  # 배치 크기 증가
        "normalize_embeddings": True,
        "convert_to_numpy": True,  # numpy 변환 최적화
        "convert_to_tensor": False  # 불필요한 텐서 변환 방지
    }
    
    embedder = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        cache_folder=".model_cache"  # 모델 캐시 폴더 지정
    )
    
    # 초기 워밍업 실행
    try:
        logging.info("임베딩 모델 워밍업 실행 중...")
        _ = embedder.embed_documents(["워밍업 텍스트"])
        logging.info("임베딩 모델 워밍업 완료")
    except Exception as e:
        logging.warning(f"워밍업 중 오류 발생: {e}")
    
    return embedder

@st.cache_data(show_spinner=False)
@log_operation("문서 분할")
def split_documents(_docs: List) -> List:
    chunker = RecursiveCharacterTextSplitter(
        chunk_size=TEXT_SPLITTER_CONFIG['chunk_size'],
        chunk_overlap=TEXT_SPLITTER_CONFIG['chunk_overlap'],
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        is_separator_regex=False,
        add_start_index=True,
    )
    return chunker.split_documents(_docs)

@st.cache_resource(show_spinner=False, ttl=600)  # 10분 후 캐시 만료
@log_operation("FAISS 벡터 저장소 생성")
def create_vector_store(_documents, _embedder, file_path: str = None) -> Optional[FAISS]:
    return FAISS.from_documents(
        documents=_documents,
        embedding=_embedder,
    )

@st.cache_resource(show_spinner=False)
@log_operation("Ollama LLM 로딩")
def load_llm(model_name: str) -> OllamaLLM:
    return OllamaLLM(
        model=model_name,
        num_predict=-1,
        )

# QA 프롬프트를 함수 외부에서 정의하여 다른 모듈에서 import 가능하게 함
QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
     """
     당신은 주어진 컨텍스트만을 사용하여 사용자의 질문에 답변하는 AI 어시스턴트입니다.
     다른 지식이나 정보를 사용해서는 안 됩니다.

     **컨텍스트**
     {context}

     **답변 생성 지침**
     1.  **언어** 사용자의 질문과 동일한 언어로 답변해야 합니다.
     2.  **답변 형식**
        - 답변은 반드시 명확하게 작성해야 합니다.
        - 답변 내용이 여러 항목, 단계 또는 문단으로 구성될 경우,
        마크다운의 줄 바꿈(예: 빈 줄 삽입)이나 목록(숫자 목록, 글머리 기호 목록)을 적절히 사용하여 가독성을 높여야 합니다.
        - 각 정보 단위가 명확히 구분되도록 표현해야 합니다.
     """
        )),
    ("human", "Question: {input}")
    ])


def update_qa_chain(llm, vector_store):
    """QA 체인 업데이트"""
    try:
        # 헬퍼 함수 정의 (st.session_state 직접 접근 제거)
        def add_doc_number_to_metadata(docs: List[Dict]) -> List[Dict]:
            """
            검색된 각 문서에 'doc_number' 메타데이터를 추가하고,
            페이지 번호를 처리합니다.
            st.session_state 접근은 이 함수 외부에서 처리합니다.
            """
            # 이 함수는 순수하게 문서 리스트를 받아 메타데이터를 추가/수정하고 반환합니다.
            # st.session_state.source_documents 관련 로직은 호출하는 쪽(메인 스레드)에서 처리합니다.
            for i, doc in enumerate(docs, 1):
                doc.metadata["doc_number"] = i # 나중에 document_prompt에서 사용

                # 페이지 번호 처리: 0-indexed를 1-indexed 문자열로 변환 또는 'N/A'
                # PyMuPDFLoader는 'page' 메타데이터를 0-indexed 정수로 제공
                page_number_raw = doc.metadata.get('page')
                if page_number_raw is not None:
                    try:
                        # Ensure page_number_raw is an integer before arithmetic operation
                        current_page_int = int(page_number_raw)
                        doc.metadata['page'] = str(current_page_int + 1) # Convert to 1-indexed string
                    except ValueError:
                        # Handle cases where page_number_raw is a string that cannot be converted to int
                        logging.warning(f"Could not convert page metadata '{page_number_raw}' to an integer. Setting page to 'N/A'.")
                        doc.metadata['page'] = 'N/A'
                else:
                    doc.metadata['page'] = 'N/A'
            return docs

        # 리트리버 설정
        faiss_retriever = vector_store.as_retriever(
            search_type=RETRIEVER_CONFIG['search_type'],
            search_kwargs=RETRIEVER_CONFIG['search_kwargs']
        )        # BM25 리트리버 설정 (분할된 문서가 세션에 저장되어 있다고 가정)
        final_retriever = faiss_retriever # 기본값은 FAISS 리트리버
        
        # 현재 파일 경로로 캐시 키 생성
        current_file = st.session_state.get('current_file_path', '')
        retriever_cache_key = f"retriever_{current_file}"
        
        if st.session_state.get("processed_document_splits"):
            try:
                # 이전 리트리버 제거
                if retriever_cache_key in st.session_state:
                    del st.session_state[retriever_cache_key]
                
                bm25_retriever = BM25Retriever.from_documents(
                    st.session_state.processed_document_splits
                )
                bm25_retriever.k = RETRIEVER_CONFIG['search_kwargs'].get('k', 5)

                final_retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, faiss_retriever],
                    weights=[0.4, 0.6] 
                )
                # 새 리트리버 캐시
                st.session_state[retriever_cache_key] = final_retriever
                logging.info(f"EnsembleRetriever (BM25 + FAISS) 생성 완료 - 파일: {current_file}")
            except Exception as e:
                logging.warning(f"BM25 리트리버 또는 EnsembleRetriever 생성 실패: {e}. FAISS 리트리버만 사용합니다.")
        else:
            logging.info("분할된 문서가 없어 FAISS 리트리버만 사용합니다.")

        # 각 문서를 LLM 프롬프트의 컨텍스트 부분에 맞게 포맷팅하기 위한 프롬프트
        # add_doc_number_to_metadata_and_save 함수에서 doc.metadata에 'doc_number'와 'page'가 설정됨
        document_prompt = PromptTemplate.from_template(
            "[{doc_number}] {page_content} (p.{page})"
        )

        # LLM에 최종적으로 전달될 프롬프트를 사용하여 문서 결합 체인 생성
        # create_stuff_documents_chain은 QA_PROMPT의 {context}를 document_prompt로 포맷된 문서들로 채움
        combine_docs_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=QA_PROMPT,
            document_prompt=document_prompt,
            document_separator="\n\n",
            document_variable_name="context" # Explicitly use "context" from QA_PROMPT
        )

        # LCEL을 사용하여 전체 RAG 체인 구성
        # 1. 세션 상태 초기화 -> 2. 입력 통과 및 문서 검색/처리 -> 3. 키 이름 변경 -> 4. LLM 호출
        # retriever가 문자열 입력을 받도록 RunnableLambda를 사용하여 'input' 키의 값을 추출
        retrieval_chain_with_processing = RunnablePassthrough.assign(
            processed_documents=RunnableLambda(lambda x: x["input"]) # 'input' 키의 값만 final_retriever로 전달
                                | final_retriever # FAISS 또는 Ensemble 리트리버
                                | RunnableLambda(add_doc_number_to_metadata) # st.session_state 접근 제거
        )

        final_qa_chain = (
            # RunnableLambda(init_source_docs_and_pass_input) # st.session_state 접근 제거
            RunnablePassthrough() # 입력: {"input": "question"}
            | retrieval_chain_with_processing # 출력: {"input": "question", "processed_documents": [docs_with_metadata]}
            # Ensure the key for documents matches what combine_docs_chain expects via QA_PROMPT's {context}
            | RunnableLambda(lambda x: {"input": x["input"], "context": x.pop("processed_documents")}) 
            # 이전 rename_documents_key 대신 context로 직접 매핑
            # | RunnableLambda(rename_documents_key) # 출력: {"input": "question", "documents": [docs_with_metadata]}
            | combine_docs_chain # 입력: {"input": "question", "documents": [docs]}, 출력: LLM 답변 문자열 (스트리밍 시 청크)
        )
        return final_qa_chain

    except Exception as e:
        raise ValueError(f"QA 체인 업데이트 실패: {e}")

def process_pdf(uploaded_file, selected_model: str, temp_pdf_path: str):
    """PDF 처리 및 QA 체인 생성."""
    try:
        # 상태 및 캐시 초기화
        st.session_state.pdf_is_processing = True
        st.session_state.pdf_processed = False
        st.session_state.qa_chain = None
        
        # 모든 캐시 초기화
        SessionManager.clear_caches()
        
        # 현재 파일 경로 설정
        st.session_state.current_file_path = temp_pdf_path
        logging.info(f"PDF 처리 시작: {temp_pdf_path}")

        # PDF 파일 저장
        docs = load_pdf_docs(temp_pdf_path)
        
        embedder = load_embedding_model()
        documents = split_documents(docs)
        st.session_state.processed_document_splits = documents # 분할된 문서 저장
        vector_store = create_vector_store(documents, embedder, temp_pdf_path)
        llm = load_llm(selected_model)
        
        # QA 체인 생성
        qa_chain = update_qa_chain(llm, vector_store)
        
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
            "- 이 문서를 한 문단으로 요약해주세요\n"
            "- 이 문서의 주요 주장과 근거를 설명해주세요\n"
            "- 이 문서의 핵심 용어 3가지를 설명해주세요\n\n"
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