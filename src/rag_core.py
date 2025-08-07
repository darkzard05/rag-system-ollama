"""
RAG 파이프라인의 핵심 로직(데이터 처리, 임베딩, 검색, 생성)을 담당하는 파일.
"""
import os
import time
import logging
import functools
import hashlib
import pickle
from typing import List, Optional, Dict, Tuple, Callable

import streamlit as st
import ollama

from session import SessionManager
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI

# --- 설정 파일에서 상수 임포트 ---
from config import (
    CACHE_DIR,
    RETRIEVER_CONFIG,
    TEXT_SPLITTER_CONFIG,
    OLLAMA_MODEL_NAME,
    GEMINI_MODEL_NAME,
    GEMINI_API_KEY,
    OLLAMA_NUM_PREDICT,
    PREFERRED_GEMINI_MODELS,
    VECTOR_STORE_CACHE_DIR
)

# --- 로깅 데코레이터 ---
def log_operation(operation_name):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logging.info(f"'{operation_name}' 시작...")
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                logging.info(f"'{operation_name}' 완료 (소요 시간: {time.time() - start_time:.2f}초)")
                return result
            except Exception as e:
                logging.error(f"'{operation_name}' 중 오류 발생: {e}", exc_info=True)
                raise
        return wrapper
    return decorator

# --- 모델 목록 조회 ---
@st.cache_data(ttl=3600)
def get_available_models() -> List[str]:
    """Ollama와 Gemini에서 사용 가능한 모델 목록을 동적으로 가져와 정렬된 리스트로 반환합니다."""
    import google.generativeai as genai
    ollama_models = []
    gemini_models = []
    
    # 1. Ollama 로컬 모델 가져오기
    try:
        ollama_response = ollama.list()
        ollama_models = sorted([
            model['model'] for model in ollama_response.get('models', [])
        ])
        if ollama_models:
            logging.info(f"Ollama에서 다음 모델을 찾았습니다: {ollama_models}")
    except Exception as e:
        logging.warning(f"Ollama 모델 목록을 가져오는 데 실패했습니다. Ollama 서버가 실행 중인지 확인하세요. 오류: {e}")

    # 2. Gemini 모델 가져오기 (선별된 최신 모델)
    if GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            available_models_from_api = [
                m.name.replace('models/', '') for m in genai.list_models() 
                if 'generateContent' in m.supported_generation_methods
            ]
            filtered_gemini_models = [
                model for model in PREFERRED_GEMINI_MODELS 
                if model in available_models_from_api
            ]

            if filtered_gemini_models:
                gemini_models = filtered_gemini_models
                logging.info(f"선별된 Gemini 모델을 찾았습니다: {gemini_models}")
            else:
                fallback_models = [
                    m for m in available_models_from_api 
                    if any(k in m for k in ["1.5", "pro"])
                ][:5]
                gemini_models = fallback_models
                logging.info(f"선호하는 Gemini 모델을 찾지 못해, 사용 가능한 모델 중 일부를 사용합니다: {fallback_models}")
        except Exception as e:
            logging.warning(f"Gemini 모델 목록을 가져오는 데 실패했습니다: {e}")

    # 3. 최종 모델 목록 조합
    final_models = []
    if ollama_models:
        final_models.extend(ollama_models)
    
    if ollama_models and gemini_models:
        final_models.append("--------------------") # 구분선 추가

    if gemini_models:
        final_models.extend(gemini_models)

    # 모델을 전혀 찾지 못한 경우 기본값 사용
    if not final_models:
        logging.error("사용 가능한 LLM 모델을 찾을 수 없습니다. 기본 모델 목록을 사용합니다.")
        return [OLLAMA_MODEL_NAME, GEMINI_MODEL_NAME]
        
    return final_models

# --- 모델 로딩 ---
@st.cache_resource(show_spinner=False)
@log_operation("임베딩 모델 로딩")
def load_embedding_model(embedding_model_name: str) -> "HuggingFaceEmbeddings":
    import torch
    from langchain_huggingface import HuggingFaceEmbeddings
    # Streamlit의 파일 감시 기능과 PyTorch 간의 호환성 문제를 해결하기 위한 임시 조치
    if hasattr(torch, 'classes'):
        torch.classes.__path__ = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"임베딩 모델용 장치: {device}")
    return HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": device},
        encode_kwargs={"device": device, "batch_size": 128},
        cache_folder=CACHE_DIR
    )

@st.cache_resource(show_spinner=False)
@log_operation("Ollama LLM 로딩")
def load_ollama_llm(_model_name: str) -> "OllamaLLM":
    from langchain_ollama import OllamaLLM
    return OllamaLLM(model=_model_name, num_predict=OLLAMA_NUM_PREDICT)

@st.cache_resource(show_spinner=False)
@log_operation("Gemini LLM 로딩")
def load_gemini_llm(_model_name: str) -> "ChatGoogleGenerativeAI":
    from langchain_google_genai import ChatGoogleGenerativeAI
    if not GEMINI_API_KEY:
        raise ValueError("config.py 파일에 Gemini API 키를 설정해야 합니다.")
    return ChatGoogleGenerativeAI(model=_model_name, google_api_key=GEMINI_API_KEY)

def load_llm(model_name: str):
    """선택된 모델 이름에 따라 적절한 LLM을 로드합니다."""
    if "gemini" in model_name.lower():
        return load_gemini_llm(_model_name=model_name)
    else:
        return load_ollama_llm(_model_name=model_name)

# --- 문서 처리 ---
@log_operation("PDF 문서 로드")
def load_pdf_docs(pdf_file_path: str) -> List:
    from langchain_community.document_loaders import PyMuPDFLoader
    if not os.path.exists(pdf_file_path):
        raise FileNotFoundError(f"PDF 파일 경로가 존재하지 않습니다: {pdf_file_path}")
    return PyMuPDFLoader(pdf_file_path).load()

@log_operation("문서 분할")
def split_documents(docs: List) -> List:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    chunker = RecursiveCharacterTextSplitter(
        chunk_size=TEXT_SPLITTER_CONFIG['chunk_size'],
        chunk_overlap=TEXT_SPLITTER_CONFIG['chunk_overlap']
    )
    return chunker.split_documents(docs)

# --- 벡터 저장소 캐싱 ---
def _get_cache_path(file_bytes: bytes, embedding_model_name: str) -> Tuple[str, str]:
    """파일 내용과 임베딩 모델 이름 기반으로 고유 캐시 경로 생성"""
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    model_name_slug = embedding_model_name.replace('/', '_')
    cache_dir = os.path.join(VECTOR_STORE_CACHE_DIR, f"{file_hash}_{model_name_slug}")
    doc_splits_path = os.path.join(cache_dir, "doc_splits.pkl")
    return cache_dir, doc_splits_path

@log_operation("벡터 저장소 캐시 저장")
def _save_to_cache(cache_path: str, doc_splits_path: str, doc_splits: List, vector_store: FAISS):
    """FAISS 인덱스와 문서 조각을 디스크에 저장"""
    try:
        os.makedirs(cache_path, exist_ok=True)
        vector_store.save_local(cache_path)
        with open(doc_splits_path, "wb") as f:
            pickle.dump(doc_splits, f)
        logging.info(f"캐시를 '{cache_path}'에 저장했습니다.")
    except Exception as e:
        logging.error(f"캐시 저장 중 오류 발생: {e}")

@log_operation("벡터 저장소 캐시 로드")
def _load_from_cache(cache_path: str, doc_splits_path: str, embedder) -> Optional[Tuple[List, FAISS]]:
    """디스크에서 FAISS 인덱스와 문서 조각을 로드"""
    if os.path.exists(cache_path) and os.path.exists(doc_splits_path):
        try:
            vector_store = FAISS.load_local(cache_path, embedder, allow_dangerous_deserialization=True)
            with open(doc_splits_path, "rb") as f:
                doc_splits = pickle.load(f)
            logging.info(f"캐시를 '{cache_path}'에서 불러왔습니다.")
            return doc_splits, vector_store
        except Exception as e:
            logging.warning(f"캐시 로드 중 오류 발생: {e}. 캐시를 재생성합니다.")
            return None
    return None

# --- 리트리버 및 벡터 저장소 생성 ---
@log_operation("FAISS 벡터 저장소 생성")
def create_vector_store(docs: List, embedder: "HuggingFaceEmbeddings") -> "FAISS":
    from langchain_community.vectorstores import FAISS
    return FAISS.from_documents(docs, embedder)

@log_operation("BM25 리트리버 생성")
def create_bm25_retriever(docs: List, k: int) -> "BM25Retriever":
    from langchain_community.retrievers import BM25Retriever
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = k
    return retriever

@log_operation("Ensemble 리트리버 생성")
def create_ensemble_retriever(faiss_retriever, bm25_retriever, weights: List[float]):
    from langchain.retrievers import EnsembleRetriever
    return EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weights=weights
    )

# --- QA 체인 구성 ---
def _add_doc_number_to_metadata(docs: List[Dict]) -> List[Dict]:
    """검색된 각 문서에 'doc_number' 메타데이터를 추가합니다."""
    for i, doc in enumerate(docs, 1):
        doc.metadata["doc_number"] = i
        page_number = doc.metadata.get('page', 'N/A')
        if page_number != 'N/A':
            doc.metadata['page'] = str(int(page_number) + 1)
    return docs

@log_operation("QA 체인 생성/업데이트")
def create_qa_chain(llm, vector_store, doc_splits: Optional[List] = None):
    """
    RAG QA 체인을 생성합니다.
    체인의 최종 출력은 {'answer': str, 'context': List[Document]} 형태가 됩니다.
    """
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough

    faiss_retriever = vector_store.as_retriever(
        search_type=RETRIEVER_CONFIG['search_type'],
        search_kwargs=RETRIEVER_CONFIG['search_kwargs']
    )
    
    if doc_splits:
        try:
            bm25_retriever = create_bm25_retriever(
                docs=doc_splits, k=RETRIEVER_CONFIG['search_kwargs']['k']
            )
            final_retriever = create_ensemble_retriever(
                faiss_retriever, bm25_retriever, RETRIEVER_CONFIG['ensemble_weights']
            )
            logging.info("EnsembleRetriever (BM25 + FAISS) 생성 및 사용.")
        except Exception as e:
            final_retriever = faiss_retriever
            logging.warning(f"EnsembleRetriever 생성 실패: {e}. FAISS 리트리버만 사용합니다.")
    else:
        final_retriever = faiss_retriever
        logging.info("분할된 문서가 없어 FAISS 리트리버만 사용합니다.")

    # 검색된 문서의 포맷을 지정하는 프롬프트
    document_prompt = PromptTemplate.from_template(
        "[{doc_number}] {page_content} (p.{page})"
    )

    # LLM에 최종적으로 전달될 프롬프트
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", (
         """
         당신은 주어진 컨텍스트만을 사용하여 사용자의 질문에 답변하는 AI 어시스턴트입니다.
         다른 지식이나 정보를 사용해서는 안 됩니다. 사용자의 질문과 동일한 언어로 답변해야 합니다.
         답변은 명확하고 가독성 높게, 필요시 마크다운 목록을 사용하여 구성해주세요.

         [컨텍스트]
         {context}
         """
        )),
        ("human", "Question: {input}")
    ])

    # 문서들을 하나의 문자열로 합치고, LLM 답변을 생성하는 체인
    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=qa_prompt,
        document_prompt=document_prompt,
        document_separator="\n\n",
    )

    # 입력부터 최종 답변까지의 전체 RAG 체인
    retrieval_chain = (
        RunnableLambda(lambda x: x["input"]) 
        | final_retriever
        | RunnableLambda(_add_doc_number_to_metadata)
    )

    final_qa_chain = (
        RunnablePassthrough.assign(
            context=retrieval_chain
        ).assign(
            answer=combine_docs_chain
        )
    )
    
    return final_qa_chain


# --- 전체 PDF 처리 파이프라인 ---
@log_operation("전체 PDF 처리 파이프라인")
def process_pdf_and_build_chain(
    uploaded_file_name: str,
    file_bytes: bytes,
    temp_pdf_path: str,
    llm,
    embedder,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Tuple[str, bool]:
    """PDF 처리부터 QA 체인 생성까지의 전체 과정을 관리합니다. 캐싱 로직 포함."""

    def _update_progress(message: str):
        if progress_callback:
            progress_callback(message)

    cache_path, doc_splits_path = _get_cache_path(file_bytes, embedder.model_name)

    # 1. 캐시 로드 시도
    _update_progress("저장된 캐시 확인 중...")
    cached_data = _load_from_cache(cache_path, doc_splits_path, embedder)

    if cached_data:
        doc_splits, vector_store = cached_data
        cache_used = True
        success_message = f"✅ '{uploaded_file_name}' 문서의 저장된 캐시를 불러왔습니다."
        _update_progress("저장된 캐시를 성공적으로 불러왔습니다.")
    else:
        # 2. 캐시가 없으면 전체 처리 수행
        cache_used = False
        _update_progress(f"'{uploaded_file_name}' 문서 로드 중...")
        docs = load_pdf_docs(temp_pdf_path)
        
        _update_progress("문서 분할 중...")
        doc_splits = split_documents(docs)
        
        _update_progress("벡터 저장소 생성 중... (시간이 소요될 수 있습니다)")
        vector_store = create_vector_store(doc_splits, embedder)

        # 3. 새로운 캐시 저장
        _update_progress("처리된 데이터 캐시 저장 중...")
        _save_to_cache(cache_path, doc_splits_path, doc_splits, vector_store)
        success_message = (
            f"✅ '{uploaded_file_name}' 문서 처리가 완료되었습니다.\n\n"
            "이제 문서 내용에 대해 자유롭게 질문해보세요."
        )

    # 4. 세션에 데이터 저장 및 QA 체인 생성
    _update_progress("QA 시스템 구성 중...")
    SessionManager.set("processed_document_splits", doc_splits)
    SessionManager.set("vector_store", vector_store)
    qa_chain = create_qa_chain(llm, vector_store, doc_splits)
    SessionManager.set("qa_chain", qa_chain)

    SessionManager.set("pdf_processed", True)
    logging.info(f"'{uploaded_file_name}' 문서 처리 및 QA 체인 생성 완료. (캐시 사용: {cache_used})")
    
    _update_progress("모든 처리가 완료되었습니다!")
    return success_message, cache_used


def is_embedding_model_cached(model_name: str) -> bool:
    """지정된 임베딩 모델이 로컬 캐시에 존재하는지 확인합니다."""
    # Hugging Face의 캐시 경로 규칙을 따릅니다.
    # 예: "sentence-transformers/all-MiniLM-L6-v2" -> ".model_cache/models--sentence-transformers--all-MiniLM-L6-v2"
    model_path_name = f"models--{model_name.replace('/', '--')}"
    cache_path = os.path.join(CACHE_DIR, model_path_name)
    return os.path.exists(cache_path)