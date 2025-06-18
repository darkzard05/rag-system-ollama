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
        "model_update_initiated_message": None, # 모델 변경 시작 메시지
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
    
    # 새 파일 업로드 시 보존할 세션 상태 키 목록
    PRESERVE_ON_NEW_FILE_KEYS = [
        "_initialized",
        "last_selected_model",
        "model_update_initiated_message",
        "last_model_change_message"
    ]
    
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
    def reset_for_new_file(cls, uploaded_file: str):
        """새 파일 업로드시 세션 상태 리셋"""
        logging.info("새 파일 업로드로 인한 세션 상태 리셋 중...")
        
        # 보존할 상태 저장
        preserved_states = {
            "model_update_initiated_message": st.session_state.get("model_update_initiated_message"),
            "last_model_change_message": st.session_state.get("last_model_change_message"),
            "last_selected_model": st.session_state.get("last_selected_model"),
            "_initialized": st.session_state.get("_initialized", False)
        }
        
        # 모든 세션 상태 초기화 (보존할 상태 제외)
        exclude_keys = ["last_selected_model", "_initialized"]
        all_keys = [key for key in st.session_state.keys() if key not in exclude_keys]
        cls.reset_session_state(all_keys)
        
        # 새 파일 정보 설정
        st.session_state.last_uploaded_file_name = uploaded_file.name
        st.session_state.current_file_path = None  # 새로운 처리 과정에서 설정됨
        
        # 1. 보존된 세션 상태 값 복원
        if preserved_states.get("model_update_initiated_message") is not None:
            st.session_state.model_update_initiated_message = preserved_states["model_update_initiated_message"]
        if preserved_states.get("last_model_change_message") is not None:
            st.session_state.last_model_change_message = preserved_states["last_model_change_message"]
        if preserved_states.get("last_selected_model") is not None:
            st.session_state.last_selected_model = preserved_states["last_selected_model"]
        st.session_state._initialized = preserved_states.get("_initialized", False)

        # 2. 보존된 메시지들을 (초기화된) messages 목록에 순서대로 다시 추가
        # 모델 변경 시작 메시지 추가
        initiated_msg = st.session_state.get("model_update_initiated_message")
        if initiated_msg:
            cls.add_message("assistant", initiated_msg)
        
        # 모델 변경 완료/결과 메시지 추가 (시작 메시지와 다를 경우에만)
        completed_msg = st.session_state.get("last_model_change_message")
        if completed_msg and completed_msg != initiated_msg:
            cls.add_message("assistant", completed_msg)
            
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
        
        model_update_msg = f"🔄 모델을 {new_model}로 변경했습니다."
        cls.add_message("assistant", model_update_msg)
        st.session_state.model_update_initiated_message = model_update_msg # 시작 메시지 저장
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
        # 이 값은 사용자의 GPU 메모리 크기 및 모델에 따라 조정될 수 있습니다.
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

@log_operation("FAISS 벡터 저장소 생성")
def create_vector_store(_documents, _embedder) -> Optional[FAISS]:
    return FAISS.from_documents(
        documents=_documents,
        embedding=_embedder,
    )
    
@log_operation("BM25 리트리버 생성")
def create_bm25_retriever(_documents: List, k: int) -> BM25Retriever:
    """캐시된 BM25 리트리버를 생성하거나 반환합니다."""
    retriever = BM25Retriever.from_documents(_documents)
    retriever.k = k
    return retriever

@log_operation("Ensemble 리트리버 생성")
def create_ensemble_retriever(_faiss_retriever: FAISS, _bm25_retriever: BM25Retriever, weights: List[float]):
    """캐시된 Ensemble 리트리버를 생성하거나 반환합니다."""
    return EnsembleRetriever(
        retrievers=[_bm25_retriever, _faiss_retriever], weights=weights
    )

@st.cache_resource(show_spinner=False)
@log_operation("Ollama LLM 로딩")
def load_llm(model_name: str) -> OllamaLLM:
    return OllamaLLM(
        model=model_name,
        num_predict=-1,
        )

# QA 체인 프롬프트 템플릿 정의
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

# 헬퍼 함수 정의 (st.session_state 직접 접근 제거)
def add_doc_number_to_metadata(docs: List[Dict]) -> List[Dict]:
    """
    검색된 각 문서에 'doc_number' 메타데이터를 추가하고,
    페이지 번호를 1-indexed 문자열로 변환합니다.
    이 함수는 순수 함수로, st.session_state에 직접 접근하지 않습니다.
    LLM 응답 후 출처를 표시하기 위해 st.session_state.source_documents 등을 활용하는 로직은
    이 함수를 호출하는 UI 측 코드에서 처리할 수 있습니다.
    """
    # 이 함수는 순수하게 문서 리스트를 받아 메타데이터를 추가/수정하고 반환합니다.
    # st.session_state.source_documents 관련 로직은 호출하는 쪽(메인 스레드)에서 처리합니다.
    for i, doc in enumerate(docs, 1):
        doc.metadata["doc_number"] = i

        # 페이지 번호 처리: 0-indexed를 1-indexed 문자열로 변환 또는 'N/A'
        # PyMuPDFLoader는 'page' 메타데이터를 0-indexed 정수로 제공
        page_number_raw = doc.metadata.get('page')
        if page_number_raw is not None:
            try:
                # 페이지 번호가 문자열로 되어 있을 경우 정수로 변환
                current_page_int = int(page_number_raw)
                doc.metadata['page'] = str(current_page_int + 1) # Convert to 1-indexed string
            except ValueError:
                # 페이지 번호가 정수로 변환 불가능한 경우
                logging.warning(f"Could not convert page metadata '{page_number_raw}' to an integer. Setting page to 'N/A'.")
                doc.metadata['page'] = 'N/A'
        else:
            doc.metadata['page'] = 'N/A'
    return docs

def update_qa_chain(llm, vector_store):
    """
    QA 체인을 업데이트합니다.
    이제 체인은 답변(answer)과 함께 참조된 문서(context)를 반환합니다.
    """
    try:
        # 1. 리트리버 설정
        faiss_retriever = vector_store.as_retriever(
            search_type=RETRIEVER_CONFIG['search_type'],
            search_kwargs=RETRIEVER_CONFIG['search_kwargs']
        )
        
        final_retriever = faiss_retriever
        
        if st.session_state.get("processed_document_splits"):
            try:
                k_val = RETRIEVER_CONFIG['search_kwargs'].get('k', 5)
                bm25_retriever_instance = create_bm25_retriever(
                    _documents=st.session_state.processed_document_splits, k=k_val
                )
                final_retriever = create_ensemble_retriever(
                    _faiss_retriever=faiss_retriever,
                    _bm25_retriever=bm25_retriever_instance,
                    weights=[0.4, 0.6]
                )
                logging.info("EnsembleRetriever (BM25 + FAISS) 생성 및 사용.")
            except Exception as e:
                logging.warning(f"EnsembleRetriever 생성 실패: {e}. FAISS 리트리버만 사용합니다.")
        else:
            logging.info("분할된 문서가 없어 FAISS 리트리버만 사용합니다.")

        # 2. 문서 포맷팅 프롬프트 (기존과 동일)
        document_prompt = PromptTemplate.from_template(
            "[{doc_number}] {page_content} (p.{page})"
        )

        # 3. LLM 호출 체인 (기존과 동일)
        combine_docs_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=QA_PROMPT,
            document_prompt=document_prompt,
            document_separator="\n\n",
        )

        # 4. LCEL을 사용하여 전체 RAG 체인 재구성 (핵심 변경 사항)
        #   - RunnablePassthrough.assign을 사용하여 체인 중간 결과를 유지합니다.
        #   - 최종 출력은 {'answer': str, 'context': List[Document]} 형태가 됩니다.

        # 4-1. 사용자의 질문(input)을 받아 문서를 검색하고 메타데이터를 추가하는 부분
        retrieval_and_metadata_chain = (
            # final_retriever는 문자열 입력을 받으므로 x['input']을 전달
            RunnableLambda(lambda x: x["input"])
            | final_retriever
            | RunnableLambda(add_doc_number_to_metadata) # 페이지 번호, 문서 번호 추가
        )

        # 4-2. 전체 체인 구성
        #   - RunnablePassthrough()로 시작하여 입력 딕셔너리({'input': '...'})를 그대로 전달
        #   - .assign(context=...) : 'context' 키에 검색된 문서를 추가
        #   - .assign(answer=...) : 'answer' 키에 LLM 답변을 추가
        final_qa_chain_with_sources = (
            RunnablePassthrough.assign(
                context=retrieval_and_metadata_chain
            ).assign(
                answer=combine_docs_chain
            )
        )
        
        return final_qa_chain_with_sources

    except Exception as e:
        # 오류 발생 시 스택 트레이스를 포함하여 더 자세한 정보 로깅
        logging.error("QA 체인 업데이트 실패", exc_info=True)
        raise ValueError(f"QA 체인 업데이트 실패: {e}")

def process_pdf(uploaded_file, selected_model: str, temp_pdf_path: str):
    """PDF 처리 및 QA 체인 생성."""
    try:
        # 상태 및 캐시 초기화
        st.session_state.pdf_is_processing = True
        st.session_state.pdf_processed = False
        st.session_state.qa_chain = None
        
        # 현재 파일 경로 설정
        st.session_state.current_file_path = temp_pdf_path
        logging.info(f"PDF 처리 시작: {temp_pdf_path}")

        # PDF 파일 저장
        docs = load_pdf_docs(temp_pdf_path)
        
        embedder = load_embedding_model()
        documents = split_documents(docs)
        st.session_state.processed_document_splits = documents # 분할된 문서 저장
        vector_store = create_vector_store(documents, embedder)
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
            "- 이 문서를 한 문단으로 요약해주세요.\n"
            "- 이 문서의 주요 주장과 근거를 설명해주세요.\n"
            "- 이 문서의 핵심 용어 3가지를 설명해주세요.\n\n"
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