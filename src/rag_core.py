"""
RAG 파이프라인의 핵심 로직(데이터 처리, 임베딩, 검색, 생성)을 담당하는 파일.
"""

import os
import logging
import hashlib
import json
from typing import List, Optional, Dict, Tuple, TYPE_CHECKING
import tempfile

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_community.vectorstores import FAISS
    from langchain_community.retrievers import BM25Retriever
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain.retrievers import EnsembleRetriever

from config import (
    RETRIEVER_CONFIG,
    TEXT_SPLITTER_CONFIG,
    VECTOR_STORE_CACHE_DIR,
)
from session import SessionManager
from utils import log_operation
from graph_builder import build_graph


# --- 문서 처리 ---
@log_operation("PDF 문서 로드")
def _load_pdf_docs(pdf_file_bytes: bytes) -> List["Document"]:
    from langchain_community.document_loaders import PyMuPDFLoader

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        temp_file.write(pdf_file_bytes)
        temp_file_path = temp_file.name

    try:
        loader = PyMuPDFLoader(file_path=temp_file_path)
        return loader.load()
    finally:
        os.remove(temp_file_path)


@log_operation("문서 분할")
def _split_documents(docs: List["Document"]) -> List["Document"]:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    chunker = RecursiveCharacterTextSplitter(
        chunk_size=TEXT_SPLITTER_CONFIG["chunk_size"],
        chunk_overlap=TEXT_SPLITTER_CONFIG["chunk_overlap"],
    )
    return chunker.split_documents(docs)


# --- 캐시 관리 ---
def _serialize_docs(docs: List["Document"]) -> List[Dict]:
    """Document 객체를 직렬화 가능한 딕셔너리 리스트로 변환합니다."""
    return [doc.dict() for doc in docs]


def _deserialize_docs(docs_as_dicts: List[Dict]) -> List["Document"]:
    """딕셔너리 리스트를 Document 객체 리스트로 복원합니다."""
    from langchain_core.documents import Document

    return [Document.parse_obj(d) for d in docs_as_dicts]


class VectorStoreCache:
    """벡터 저장소 캐시를 관리(경로 생성, 저장, 로드)합니다."""

    def __init__(self, file_bytes: bytes, embedding_model_name: str):
        self.cache_dir, self.doc_splits_path, self.faiss_index_path = (
            self._get_cache_paths(file_bytes, embedding_model_name)
        )

    def _get_cache_paths(
        self, file_bytes: bytes, embedding_model_name: str
    ) -> Tuple[str, str, str]:
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        model_name_slug = embedding_model_name.replace("/", "_")
        cache_dir = os.path.join(
            VECTOR_STORE_CACHE_DIR, f"{file_hash}_{model_name_slug}"
        )
        doc_splits_path = os.path.join(cache_dir, "doc_splits.json")
        faiss_index_path = os.path.join(cache_dir, "faiss_index")
        return cache_dir, doc_splits_path, faiss_index_path

    def load(
        self, embedder: "HuggingFaceEmbeddings"
    ) -> Tuple[Optional[List["Document"]], Optional["FAISS"]]:
        from langchain_community.vectorstores import FAISS

        if not all(
            os.path.exists(p) for p in [self.doc_splits_path, self.faiss_index_path]
        ):
            return None, None
        try:
            with open(self.doc_splits_path, "r", encoding="utf-8") as f:
                doc_splits = _deserialize_docs(json.load(f))

            vector_store = FAISS.load_local(
                self.faiss_index_path,
                embedder,
                allow_dangerous_deserialization=True,
            )
            logging.info(f"벡터 저장소 캐시를 '{self.cache_dir}'에서 불러왔습니다.")
            return doc_splits, vector_store
        except Exception as e:
            logging.warning(f"캐시 로드 중 오류 발생: {e}. 캐시를 재생성합니다.")
            return None, None

    def save(self, doc_splits: List["Document"], vector_store: "FAISS"):
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(self.doc_splits_path, "w", encoding="utf-8") as f:
                json.dump(_serialize_docs(doc_splits), f, ensure_ascii=False, indent=4)
            vector_store.save_local(self.faiss_index_path)
            logging.info(f"벡터 저장소 캐시를 '{self.cache_dir}'에 저장했습니다.")
        except Exception as e:
            logging.error(f"캐시 저장 중 오류 발생: {e}")


# --- 리트리버 생성 ---
@log_operation("FAISS 벡터 저장소 및 리트리버 생성")
def _create_vector_store(
    docs: List["Document"], embedder: "HuggingFaceEmbeddings"
) -> "FAISS":
    from langchain_community.vectorstores import FAISS

    return FAISS.from_documents(docs, embedder)


@log_operation("BM25 리트리버 생성")
def _create_bm25_retriever(
    docs: List["Document"], k: int
) -> "BM25Retriever":
    from langchain_community.retrievers import BM25Retriever

    retriever = BM25Retriever.from_documents(docs)
    retriever.k = k
    return retriever


@log_operation("Ensemble 리트리버 생성")
def _create_ensemble_retriever(
    faiss_retriever, bm25_retriever, weights: List[float]
) -> "EnsembleRetriever":
    from langchain.retrievers import EnsembleRetriever

    return EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weights=weights
    )


# --- 파이프라인 구축 ---
@log_operation("문서 로드 및 분할 (캐시 확인)")
def _load_or_create_splits(
    file_bytes: bytes, embedder: "HuggingFaceEmbeddings"
) -> Tuple[List["Document"], "FAISS", bool]:
    cache = VectorStoreCache(file_bytes, embedder.model_name)
    doc_splits, vector_store = cache.load(embedder)
    cache_used = bool(doc_splits and vector_store)

    if not cache_used:
        docs = _load_pdf_docs(file_bytes)
        doc_splits = _split_documents(docs)
        vector_store = _create_vector_store(doc_splits, embedder)
        cache.save(doc_splits, vector_store)

    return doc_splits, vector_store, cache_used


def _create_retriever(
    doc_splits: List["Document"], vector_store: "FAISS"
) -> "EnsembleRetriever":
    faiss_retriever = vector_store.as_retriever(
        search_type=RETRIEVER_CONFIG["search_type"],
        search_kwargs=RETRIEVER_CONFIG["search_kwargs"],
    )
    bm25_retriever = _create_bm25_retriever(
        docs=doc_splits, k=RETRIEVER_CONFIG["search_kwargs"]["k"]
    )
    return _create_ensemble_retriever(
        faiss_retriever, bm25_retriever, RETRIEVER_CONFIG["ensemble_weights"]
    )


@log_operation("RAG 파이프라인 구축")
def build_rag_pipeline(
    uploaded_file_name: str, file_bytes: bytes, embedder: "HuggingFaceEmbeddings"
) -> Tuple[str, bool]:
    doc_splits, vector_store, cache_used = _load_or_create_splits(
        file_bytes, embedder
    )
    final_retriever = _create_retriever(doc_splits, vector_store)
    rag_app = build_graph(retriever=final_retriever)

    SessionManager.set("processed_document_splits", doc_splits)
    SessionManager.set("qa_chain", rag_app)
    SessionManager.set("pdf_processed", True)

    logging.info(
        f"'{uploaded_file_name}' 문서 처리 및 LangGraph 기반 QA 체인 생성 완료. (캐시 사용: {cache_used})"
    )

    if cache_used:
        return f"✅ '{uploaded_file_name}' 문서의 저장된 캐시를 불러왔습니다.", True
    return (
        f"✅ '{uploaded_file_name}' 문서 처리가 완료되었습니다.\n\n"
        "이제 문서 내용에 대해 자유롭게 질문해보세요."
    ), False


@log_operation("파이프라인의 LLM 업데이트")
def update_llm_in_pipeline(llm):
    """세션의 LLM을 교체합니다. 그래프가 세션에서 LLM을 가져오므로 재빌드할 필요가 없습니다."""
    if not SessionManager.get("pdf_processed"):
        raise ValueError("RAG 파이프라인이 구축되지 않아 LLM을 업데이트할 수 없습니다.")

    SessionManager.set("llm", llm)
    logging.info(f"세션의 LLM이 새로운 모델 '{llm.model}'(으)로 업데이트되었습니다.")
