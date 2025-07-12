"""
RAG 파이프라인의 핵심 로직(데이터 처리, 임베딩, 검색, 생성)을 담당하는 파일.
"""
import os
import time
import logging
import functools
from typing import List, Optional, Dict

import torch
# Streamlit의 파일 감시 기능과 PyTorch 간의 호환성 문제를 해결하기 위한 임시 조치
if not hasattr(torch.classes, '__path__'):
    torch.classes.__path__ = []

import ollama
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
from tqdm import tqdm

# 설정 파일에서 상수 임포트
from config import (
    EMBEDDING_MODEL_NAME,
    CACHE_DIR,
    RETRIEVER_CONFIG,
    TEXT_SPLITTER_CONFIG
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

# --- 외부 서비스 연동 ---
@st.cache_data(show_spinner=False)
@log_operation("Ollama 모델 목록 불러오기")
def get_ollama_models() -> List[str]:
    response = ollama.list()
    if 'models' in response:
        return [model['model'] for model in response['models']]
    logging.warning("Ollama 응답에 'models' 키가 없습니다. 빈 목록을 반환합니다.")
    return []

# --- 모델 로딩 ---
@st.cache_resource(show_spinner=False)
@log_operation("임베딩 모델 로딩")
def load_embedding_model() -> HuggingFaceEmbeddings:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"임베딩 모델용 장치: {device}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device},
        encode_kwargs={"device": device, "batch_size": 128},
        cache_folder=CACHE_DIR
    )

@st.cache_resource(show_spinner=False)
@log_operation("Ollama LLM 로딩")
def load_llm(model_name: str) -> OllamaLLM:
    return OllamaLLM(model=model_name, num_predict=-1)

# --- 문서 처리 ---
@log_operation("PDF 문서 로드")
def load_pdf_docs(pdf_file_path: str) -> List:
    if not os.path.exists(pdf_file_path):
        raise FileNotFoundError(f"PDF 파일 경로가 존재하지 않습니다: {pdf_file_path}")
    return PyMuPDFLoader(pdf_file_path).load()

@log_operation("문서 분할")
def split_documents(docs: List) -> List:
    chunker = RecursiveCharacterTextSplitter(
        chunk_size=TEXT_SPLITTER_CONFIG['chunk_size'],
        chunk_overlap=TEXT_SPLITTER_CONFIG['chunk_overlap']
    )
    return chunker.split_documents(docs)

# --- 리트리버 및 벡터 저장소 생성 ---
@log_operation("FAISS 벡터 저장소 생성")
def create_vector_store(docs: List, embedder: HuggingFaceEmbeddings) -> FAISS:
    return FAISS.from_documents(docs, embedder)

@log_operation("BM25 리트리버 생성")
def create_bm25_retriever(docs: List, k: int) -> BM25Retriever:
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = k
    return retriever

@log_operation("Ensemble 리트리버 ��성")
def create_ensemble_retriever(faiss_retriever, bm25_retriever, weights: List[float]):
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
                faiss_retriever, bm25_retriever, RETRIEVER_CONFIG['weights']
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
         
         답변을 생성하기 전에, 먼저 당신의 생각 과정을 <think>...</think> 태그 안에 정리해주세요.
         예: <think>사용자의 질문은 A에 대한 것이다. 컨텍스트 2, 5에서 관련 정보를 찾았다. 이 정보를 종합하여 답변을 구성해야겠다.</think>

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
def process_pdf_and_build_chain(uploaded_file, temp_pdf_path: str, selected_model: str):
    """PDF 처리부터 QA 체인 생성까지의 전체 과정을 관리합니다."""
    
    # 1. 문서 로드 및 분할
    docs = load_pdf_docs(temp_pdf_path)
    doc_splits = split_documents(docs)
    st.session_state.processed_document_splits = doc_splits
    
    # 2. 임베딩 및 벡터 저장소 생성
    embedder = load_embedding_model()
    vector_store = create_vector_store(doc_splits, embedder)
    st.session_state.vector_store = vector_store
    
    # 3. LLM 로드 및 QA 체인 생성
    llm = load_llm(selected_model)
    st.session_state.llm = llm
    qa_chain = create_qa_chain(llm, vector_store, doc_splits)
    st.session_state.qa_chain = qa_chain
    
    st.session_state.pdf_processed = True
    logging.info(f"'{uploaded_file.name}' 문서 처리 및 QA 체인 생성 완료.")
    
    success_message = (
        f"✅ '{uploaded_file.name}' 문서 처리가 완료되었습니다.\n\n"
        "이제 문서 내용에 대해 자유롭게 질문해보세요."
    )
    return success_message
