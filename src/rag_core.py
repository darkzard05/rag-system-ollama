"""
RAG 파이프라인의 핵심 로직(데이터 처리, 임베딩, 검색, 생성)을 담당하는 파일.
"""

import os
import logging
import hashlib
import json
from typing import List, Optional, Dict, Tuple
import tempfile

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever

# --- 설정 파일 및 모듈 임포트 ---
from config import (
    RETRIEVER_CONFIG,
    TEXT_SPLITTER_CONFIG,
    VECTOR_STORE_CACHE_DIR,
)
from session import SessionManager
from utils import log_operation
from graph_builder import build_graph # LangGraph 빌더 임포트


# --- 문서 처리 ---
@log_operation("PDF 문서 로드")
def load_pdf_docs(pdf_file_bytes: bytes) -> List:
    """PDF 파일 바이트에서 문서를 로드합니다."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        temp_file.write(pdf_file_bytes)
        temp_file_path = temp_file.name
    
    try:
        loader = PyMuPDFLoader(file_path=temp_file_path)
        docs = loader.load()
    finally:
        os.remove(temp_file_path)

    return docs


@log_operation("문서 분할")
def split_documents(docs: List) -> List:
    chunker = RecursiveCharacterTextSplitter(
        chunk_size=TEXT_SPLITTER_CONFIG["chunk_size"],
        chunk_overlap=TEXT_SPLITTER_CONFIG["chunk_overlap"],
    )
    return chunker.split_documents(docs)


# --- 벡터 저장소 캐싱 ---
class VectorStoreCache:
    """벡터 저장소 캐싱을 관리하는 클래스"""

    def __init__(self, file_bytes: bytes, embedding_model_name: str):
        self.cache_dir, self.doc_splits_path = self._get_cache_paths(
            file_bytes, embedding_model_name
        )

    def _get_cache_paths(
        self, file_bytes: bytes, embedding_model_name: str
    ) -> Tuple[str, str]:
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        model_name_slug = embedding_model_name.replace("/", "_")
        cache_dir = os.path.join(
            VECTOR_STORE_CACHE_DIR, f"{file_hash}_{model_name_slug}"
        )
        doc_splits_path = os.path.join(cache_dir, "doc_splits.json")
        return cache_dir, doc_splits_path

    def _serialize_docs(self, docs: List[Document]) -> List[Dict]:
        return [
            {"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs
        ]

    def _deserialize_docs(self, docs_as_dicts: List[Dict]) -> List[Document]:
        return [
            Document(page_content=d["page_content"], metadata=d["metadata"])
            for d in docs_as_dicts
        ]

    @log_operation("문서 조각 캐시 로드")
    def load(self) -> Optional[List[Document]]:
        if os.path.exists(self.doc_splits_path):
            try:
                with open(self.doc_splits_path, "r", encoding="utf-8") as f:
                    doc_splits_as_dicts = json.load(f)
                doc_splits = self._deserialize_docs(doc_splits_as_dicts)
                logging.info(f"문서 조각 캐시를 '{self.doc_splits_path}'에서 불러왔습니다.")
                return doc_splits
            except Exception as e:
                logging.warning(f"문서 조각 캐시 로드 중 오류 발생: {e}. 캐시를 재생성합니다.")
        return None

    @log_operation("문서 조각 캐시 저장")
    def save(self, doc_splits: List):
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(self.doc_splits_path, "w", encoding="utf-8") as f:
                json.dump(
                    self._serialize_docs(doc_splits), f, ensure_ascii=False, indent=4
                )
            logging.info(f"문서 조각 캐시를 '{self.doc_splits_path}'에 저장했습니다.")
        except Exception as e:
            logging.error(f"문서 조각 캐시 저장 중 오류 발생: {e}")


# --- 리트리버 및 벡터 저장소 생성 ---
@log_operation("FAISS 벡터 저장소 생성")
def create_vector_store(docs: List, embedder: "HuggingFaceEmbeddings") -> "FAISS":
    return FAISS.from_documents(docs, embedder)


@log_operation("BM25 리트리버 생성")
def create_bm25_retriever(docs: List, k: int) -> "BM25Retriever":
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = k
    return retriever


@log_operation("Ensemble 리트리버 생성")
def create_ensemble_retriever(faiss_retriever, bm25_retriever, weights: List[float]):
    return EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weights=weights
    )


# --- 전체 PDF 처리 및 RAG 파이프라인 구축 ---
@log_operation("RAG 파이프라인 구축")
def build_rag_pipeline(
    uploaded_file_name: str, file_bytes: bytes, llm, embedder
) -> Tuple[str, bool]:
    """세션을 업데이트하고 RAG 파이프라인 전체를 구축합니다."""
    
    # 1. 문서 처리 및 벡터 저장소 생성 (캐싱 로직 포함)
    cache = VectorStoreCache(file_bytes, embedder.model_name)
    doc_splits = cache.load()
    cache_used = False

    if doc_splits:
        cache_used = True
        vector_store = create_vector_store(doc_splits, embedder)
    else:
        docs = load_pdf_docs(file_bytes)
        doc_splits = split_documents(docs)
        vector_store = create_vector_store(doc_splits, embedder)
        cache.save(doc_splits)

    # 2. 리트리버 생성
    faiss_retriever = vector_store.as_retriever(
        search_type=RETRIEVER_CONFIG["search_type"],
        search_kwargs=RETRIEVER_CONFIG["search_kwargs"],
    )
    bm25_retriever = create_bm25_retriever(
        docs=doc_splits, k=RETRIEVER_CONFIG["search_kwargs"]["k"]
    )
    final_retriever = create_ensemble_retriever(
        faiss_retriever, bm25_retriever, RETRIEVER_CONFIG["ensemble_weights"]
    )

    # 3. LangGraph 앱 빌드
    rag_app = build_graph() # LangGraph를 사용하여 RAG 앱을 빌드합니다.

    # 4. 세션 업데이트
    SessionManager.set("processed_document_splits", doc_splits)
    SessionManager.set("retriever", final_retriever) # 그래프 노드에서 사용할 리트리버 저장
    SessionManager.set("qa_chain", rag_app) # 컴파일된 그래프 앱을 세션에 저장
    SessionManager.set("pdf_processed", True)

    logging.info(
        f"'{uploaded_file_name}' 문서 처리 및 LangGraph 기반 QA 체인 생성 완료. (캐시 사용: {cache_used})"
    )

    if cache_used:
        success_message = f"✅ '{uploaded_file_name}' 문서의 저장된 캐시를 불러왔습니다."
    else:
        success_message = (
            f"✅ '{uploaded_file_name}' 문서 처리가 완료되었습니다.\n\n"
            "이제 문서 내용에 대해 자유롭게 질문해보세요."
        )
    return success_message, cache_used


@log_operation("파이프라인의 LLM 업데이트")
def update_llm_in_pipeline(llm):
    """세션의 LLM을 교체합니다. LangGraph는 실행 시점에 세션에서 LLM을 가져오므로 그래프를 재빌드할 필요가 없습니다."""
    if not SessionManager.get("pdf_processed"):
        raise ValueError("RAG 파이IP라인이 구축되지 않아 LLM을 업데이트할 수 없습니다.")

    SessionManager.set("llm", llm)
    logging.info(f"세션의 LLM이 새로운 모델 '{llm.model}'(으)로 업데이트되었습니다.")
